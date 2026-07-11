from dataclasses import dataclass
import math
from pathlib import Path
import random

import torch
import torch.nn as nn
from torch.nn import functional as F

from model.plm import ProteinLM, ProteinLMConfig
from model.protein_tokenizer import ESM_PAD_ID, RES_PAD_ID, ProteinTokenizer


ATOM_SLOTS = (
    'N', 'CA', 'C', 'O', 'CB', 'CG', 'CG1', 'CG2', 'OG', 'OG1', 'SG', 'CD',
    'CD1', 'CD2', 'ND1', 'ND2', 'OD1', 'OD2', 'SD', 'CE', 'CE1', 'CE2',
    'CE3', 'NE', 'NE1', 'NE2', 'OE1', 'OE2', 'CH2', 'NH1', 'NH2', 'OH',
    'CZ', 'CZ2', 'CZ3', 'NZ', 'OXT',
)
ATOM_SLOT_TO_INDEX = {name: index for index, name in enumerate(ATOM_SLOTS)}
ATOM_NAME_CHAR_WIDTH = 4
ATOM_ELEMENT_VOCAB_SIZE = 128
ATOM_NAME_CHAR_VOCAB_SIZE = 64
MAX_ATOMS_PER_TOKEN = len(ATOM_SLOTS)


def sequence_pair_mask(residue_mask):
    residue_mask = residue_mask.bool()
    return residue_mask[:, :, None] & residue_mask[:, None, :]


def coordinate_pair_mask(coord_mask, remove_diagonal=False):
    coord_mask = coord_mask.bool()
    pair_mask = coord_mask[:, :, None] & coord_mask[:, None, :]
    if remove_diagonal:
        length = coord_mask.size(1)
        diagonal = torch.eye(length, dtype=torch.bool, device=coord_mask.device)
        pair_mask = pair_mask & ~diagonal[None, :, :]
    return pair_mask


def masked_center(coords, coord_mask):
    coordinate_values = coord_mask.to(dtype=coords.dtype)[..., None]
    center_dims = tuple(range(1, coords.ndim - 1))
    coordinate_count = coordinate_values.sum(
        dim=center_dims,
        keepdim=True,
    ).clamp_min(1.0)
    coordinate_center = (
        coords * coordinate_values
    ).sum(dim=center_dims, keepdim=True) / coordinate_count
    return (coords - coordinate_center) * coordinate_values


def scatter_atom_to_token(atom_features, atom_to_token, token_count, atom_mask=None):
    batch, atom_count, feature_dim = atom_features.shape
    if atom_mask is None:
        atom_mask = atom_to_token >= 0
    valid_atom_to_token = torch.where(
        atom_mask.bool(),
        atom_to_token.clamp_min(0),
        torch.full_like(atom_to_token, token_count),
    )
    output = atom_features.new_zeros(batch, token_count + 1, feature_dim)
    counts = atom_features.new_zeros(batch, token_count + 1, 1)
    output.scatter_add_(
        1,
        valid_atom_to_token[..., None].expand(batch, atom_count, feature_dim),
        atom_features * atom_mask.to(atom_features.dtype)[..., None],
    )
    counts.scatter_add_(
        1,
        valid_atom_to_token[..., None],
        atom_mask.to(atom_features.dtype)[..., None],
    )
    return output[:, :token_count] / counts[:, :token_count].clamp_min(1.0)


def gather_token_to_atom(token_features, atom_to_token):
    index = atom_to_token.clamp_min(0)
    index = index[..., None].expand(-1, -1, token_features.size(-1))
    return token_features.gather(dim=1, index=index)


def compute_intra_token_index(atom_to_token, atom_mask):
    intra_index = torch.zeros_like(atom_to_token)
    token_count = int(atom_to_token.clamp_min(0).max().item()) + 1
    for token_index in range(token_count):
        token_atoms = (atom_to_token == token_index) & atom_mask
        running_index = torch.cumsum(token_atoms.long(), dim=1) - 1
        intra_index = torch.where(token_atoms, running_index, intra_index)
    return intra_index.clamp(min=0, max=MAX_ATOMS_PER_TOKEN - 1)


def gather_rep_atom_coords(atom_coords, distogram_atom_idx):
    index = distogram_atom_idx.clamp_min(0)[..., None].expand(-1, -1, 3)
    return atom_coords.gather(dim=1, index=index)


def gather_rep_atom_mask(atom_mask, distogram_atom_idx):
    index = distogram_atom_idx.clamp_min(0)
    return atom_mask.bool().gather(dim=1, index=index) & (distogram_atom_idx >= 0)


def validate_ref_space_uid(ref_space_uid, atom_mask, max_uid):
    valid_ref_spaces = atom_mask.bool()
    invalid = valid_ref_spaces & ((ref_space_uid < 0) | (ref_space_uid > max_uid))
    if invalid.any():
        raise ValueError('ref_space_uid values for valid atoms must be in [0, block_size]')


def pairwise_distances(coords):
    residue_deltas = coords[:, :, None, :] - coords[:, None, :, :]
    distance_squared = (residue_deltas * residue_deltas).sum(dim=-1)
    return torch.sqrt(distance_squared.clamp_min(0.0) + 1e-8)


def masked_mean(values, mask):
    expanded_mask = mask.to(dtype=values.dtype)
    while expanded_mask.ndim < values.ndim:
        expanded_mask = expanded_mask.unsqueeze(-1)
    expanded_mask = expanded_mask.expand_as(values)
    numerator = (values * expanded_mask).sum()
    denominator = expanded_mask.sum().clamp_min(1.0)
    return numerator / denominator


def add_atom_token_fields(batch):
    atom_coords_slots = batch['atom_coords']
    atom_mask_slots = batch['atom_mask'].bool()
    reference_atom_coords_slots = batch['reference_atom_coords']
    reference_atom_mask_slots = batch['reference_atom_mask'].bool()
    atom_to_token_slots = batch['atom_to_token']
    batch_size, residue_length, atom_count = atom_mask_slots.shape
    device = atom_mask_slots.device

    positions = torch.arange(residue_length, device=device, dtype=torch.long)
    positions = positions.view(1, residue_length).expand(batch_size, residue_length)
    zeros = torch.zeros_like(positions)

    if 'token_bonds' in batch and batch['token_bonds'].ndim == 3:
        token_bonds = batch['token_bonds'].bool()
    else:
        token_bonds = torch.zeros(
            batch_size,
            residue_length,
            residue_length,
            dtype=torch.bool,
            device=device,
        )
    flat_atom_to_token = atom_to_token_slots.reshape(batch_size, residue_length * atom_count).long()
    flat_reference_mask = reference_atom_mask_slots.reshape(batch_size, residue_length * atom_count)
    invalid_token_map = flat_reference_mask & (
        (flat_atom_to_token < 0) | (flat_atom_to_token >= residue_length)
    )
    if invalid_token_map.any():
        raise ValueError('atom_to_token must use local residue indices for the current crop')
    atom_bonds = torch.zeros(
        batch_size,
        residue_length * atom_count,
        residue_length * atom_count,
        dtype=torch.bool,
        device=device,
    )
    if 'residue_atom_bonds' in batch:
        residue_atom_bonds = batch['residue_atom_bonds'].bool()
        residue_atom_bonds = (
            residue_atom_bonds
            & reference_atom_mask_slots[:, :, :, None]
            & reference_atom_mask_slots[:, :, None, :]
        )
        for residue_index in range(residue_length):
            atom_start = residue_index * atom_count
            atom_end = atom_start + atom_count
            atom_bonds[:, atom_start:atom_end, atom_start:atom_end] = (
                residue_atom_bonds[:, residue_index]
            )
    if 'peptide_bond_mask' in batch and residue_length > 1:
        c_index = ATOM_SLOT_TO_INDEX['C']
        n_index = ATOM_SLOT_TO_INDEX['N']
        peptide_bonds = batch['peptide_bond_mask'].bool()
        for residue_index in range(residue_length - 1):
            atom_a = residue_index * atom_count + c_index
            atom_b = (residue_index + 1) * atom_count + n_index
            valid_bond = (
                peptide_bonds[:, residue_index]
                & reference_atom_mask_slots[:, residue_index, c_index]
                & reference_atom_mask_slots[:, residue_index + 1, n_index]
            )
            atom_bonds[:, atom_a, atom_b] = valid_bond
            atom_bonds[:, atom_b, atom_a] = valid_bond
    if 'ref_space_uid' in batch and batch['ref_space_uid'].ndim == 3:
        ref_space_uid = batch['ref_space_uid'].reshape(batch_size, residue_length * atom_count).long()
    else:
        ref_space_uid = torch.where(
            flat_reference_mask,
            flat_atom_to_token + 1,
            torch.zeros_like(flat_atom_to_token),
        )
    cb_index = ATOM_SLOT_TO_INDEX['CB']
    ca_index = ATOM_SLOT_TO_INDEX['CA']
    cb_atom_idx = positions * atom_count + cb_index
    ca_atom_idx = positions * atom_count + ca_index
    first_valid_atom = reference_atom_mask_slots.float().argmax(dim=-1)
    has_cb = reference_atom_mask_slots[:, :, cb_index]
    has_ca = reference_atom_mask_slots[:, :, ca_index]
    distogram_atom_idx = torch.where(
        has_cb,
        cb_atom_idx,
        torch.where(
            has_ca,
            ca_atom_idx,
            positions * atom_count + first_valid_atom,
        ),
    )

    batch['atom_coords_slots'] = atom_coords_slots
    batch['atom_mask_slots'] = atom_mask_slots
    batch['reference_atom_coords_slots'] = reference_atom_coords_slots
    batch['reference_atom_mask_slots'] = reference_atom_mask_slots
    batch['atom_to_token_slots'] = atom_to_token_slots
    batch['atom_coords'] = atom_coords_slots.reshape(batch_size, residue_length * atom_count, 3)
    batch['atom_mask'] = atom_mask_slots.reshape(batch_size, residue_length * atom_count)
    batch['ref_pos'] = reference_atom_coords_slots.reshape(batch_size, residue_length * atom_count, 3)
    batch['atom_attention_mask'] = flat_reference_mask
    batch['ref_space_uid'] = ref_space_uid
    batch['atom_to_token'] = flat_atom_to_token
    batch['atom_bonds'] = atom_bonds
    batch['atom_element'] = batch['atom_element'].reshape(batch_size, residue_length * atom_count).long()
    batch['atom_charge'] = batch['atom_charge'].reshape(batch_size, residue_length * atom_count)
    batch['atom_name_chars'] = batch['atom_name_chars'].reshape(
        batch_size,
        residue_length * atom_count,
        ATOM_NAME_CHAR_WIDTH,
    ).long()
    batch['token_bonds'] = token_bonds
    batch['token_attention_mask'] = batch['residue_mask']
    batch['distogram_atom_idx'] = distogram_atom_idx
    batch['residue_index'] = batch.get('residue_index', positions)
    batch['token_index'] = batch.get('token_index', positions)
    batch['asym_id'] = batch.get('asym_id', zeros)
    batch['sym_id'] = batch.get('sym_id', zeros)
    batch['entity_id'] = batch.get('entity_id', zeros)
    batch['mol_type'] = batch.get('mol_type', zeros)
    return batch


def localize_atom_token_indices(example, crop_start):
    if crop_start == 0:
        return example
    if 'atom_to_token' in example:
        atom_to_token = example['atom_to_token']
        valid_atoms = atom_to_token >= 0
        example['atom_to_token'] = torch.where(
            valid_atoms,
            atom_to_token - crop_start,
            atom_to_token,
        )
    if 'ref_space_uid' in example:
        ref_space_uid = example['ref_space_uid']
        valid_spaces = ref_space_uid > 0
        example['ref_space_uid'] = torch.where(
            valid_spaces,
            ref_space_uid - crop_start,
            ref_space_uid,
        ).clamp_min(0)
    return example


def centered_gaussian_noise(reference_coords, coord_mask, generator=None):
    noise = reference_coords.new_empty(reference_coords.shape)
    noise.normal_(generator=generator)
    return masked_center(noise, coord_mask)


def representative_atom_coords(atom_coords, atom_mask):
    ca_index = ATOM_SLOT_TO_INDEX['CA']
    ca_mask = atom_mask[:, :, ca_index]
    if ca_mask.any():
        return atom_coords[:, :, ca_index], ca_mask

    atom_values = atom_mask.to(atom_coords.dtype)[..., None]
    atom_count = atom_values.sum(dim=2).clamp_min(1.0)
    coords = (atom_coords * atom_values).sum(dim=2) / atom_count
    return coords, atom_mask.any(dim=2)


def atom_lddt(predicted_atom_coords, true_atom_coords, atom_mask, cutoff=15.0):
    batch_size, residue_length, atom_count, _ = predicted_atom_coords.shape
    predicted = predicted_atom_coords.reshape(batch_size, residue_length * atom_count, 3)
    target = true_atom_coords.reshape(batch_size, residue_length * atom_count, 3)
    flat_mask = atom_mask.reshape(batch_size, residue_length * atom_count)

    predicted_distances = pairwise_distances(predicted)
    true_distances = pairwise_distances(target)
    pair_mask = flat_mask[:, :, None] & flat_mask[:, None, :]
    diagonal = torch.eye(pair_mask.size(1), dtype=torch.bool, device=pair_mask.device)
    pair_mask = pair_mask & ~diagonal[None, :, :] & (true_distances < cutoff)
    distance_error = (predicted_distances - true_distances).abs()

    score_sum = torch.zeros_like(distance_error)
    for threshold in (0.5, 1.0, 2.0, 4.0):
        score_sum = score_sum + (distance_error < threshold).to(distance_error.dtype)
    pair_score = score_sum / 4.0

    pair_values = pair_mask.to(pair_score.dtype)
    atom_neighbor_count = pair_values.sum(dim=-1)
    atom_score = (
        pair_score * pair_values
    ).sum(dim=-1) / atom_neighbor_count.clamp_min(1.0)
    atom_score = atom_score * (atom_neighbor_count > 0).to(atom_score.dtype)
    atom_score = atom_score.reshape(batch_size, residue_length, atom_count)

    atom_values = atom_mask.to(atom_score.dtype)
    atom_count_per_residue = atom_values.sum(dim=-1).clamp_min(1.0)
    residue_score = (atom_score * atom_values).sum(dim=-1) / atom_count_per_residue
    return residue_score * atom_mask.any(dim=-1).to(residue_score.dtype)


def atom_lddt_per_atom(predicted_atom_coords, true_atom_coords, atom_mask, cutoff=15.0):
    batch_size, residue_length, atom_count, _ = predicted_atom_coords.shape
    predicted = predicted_atom_coords.reshape(batch_size, residue_length * atom_count, 3)
    target = true_atom_coords.reshape(batch_size, residue_length * atom_count, 3)
    flat_mask = atom_mask.reshape(batch_size, residue_length * atom_count)

    predicted_distances = pairwise_distances(predicted)
    true_distances = pairwise_distances(target)
    pair_mask = flat_mask[:, :, None] & flat_mask[:, None, :]
    diagonal = torch.eye(pair_mask.size(1), dtype=torch.bool, device=pair_mask.device)
    pair_mask = pair_mask & ~diagonal[None, :, :] & (true_distances < cutoff)
    distance_error = (predicted_distances - true_distances).abs()

    score_sum = torch.zeros_like(distance_error)
    for threshold in (0.5, 1.0, 2.0, 4.0):
        score_sum = score_sum + (distance_error < threshold).to(distance_error.dtype)
    pair_score = score_sum / 4.0

    pair_values = pair_mask.to(pair_score.dtype)
    atom_neighbor_count = pair_values.sum(dim=-1)
    atom_score = (
        pair_score * pair_values
    ).sum(dim=-1) / atom_neighbor_count.clamp_min(1.0)
    atom_score = atom_score * (atom_neighbor_count > 0).to(atom_score.dtype)
    return atom_score.reshape(batch_size, residue_length, atom_count)


def atom_lddt_per_atom_flat(predicted_atom_coords, true_atom_coords, atom_mask, cutoff=15.0):
    predicted_distances = pairwise_distances(predicted_atom_coords)
    true_distances = pairwise_distances(true_atom_coords)
    pair_mask = atom_mask[:, :, None] & atom_mask[:, None, :]
    diagonal = torch.eye(pair_mask.size(1), dtype=torch.bool, device=pair_mask.device)
    pair_mask = pair_mask & ~diagonal[None, :, :] & (true_distances < cutoff)
    distance_error = (predicted_distances - true_distances).abs()

    score_sum = torch.zeros_like(distance_error)
    for threshold in (0.5, 1.0, 2.0, 4.0):
        score_sum = score_sum + (distance_error < threshold).to(distance_error.dtype)
    pair_score = score_sum / 4.0

    pair_values = pair_mask.to(pair_score.dtype)
    atom_neighbor_count = pair_values.sum(dim=-1)
    atom_score = (
        pair_score * pair_values
    ).sum(dim=-1) / atom_neighbor_count.clamp_min(1.0)
    atom_score = atom_score * (atom_neighbor_count > 0).to(atom_score.dtype)
    return atom_score * atom_mask.to(atom_score.dtype)


def random_rotation_matrix(reference_coords, generator=None):
    batch_size = reference_coords.size(0)
    matrices = reference_coords.new_empty(batch_size, 3, 3)
    matrices.normal_(generator=generator)
    orthogonal, upper_triangular = torch.linalg.qr(matrices)
    signs = torch.sign(torch.diagonal(upper_triangular, dim1=-2, dim2=-1))
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    rotation = orthogonal * signs[:, None, :]
    determinant = torch.linalg.det(rotation)
    rotation[:, :, -1] = rotation[:, :, -1] * determinant[:, None]
    return rotation


def make_sampling_schedule(reference, sigma_min, sigma_max, num_steps, rho=7.0):
    assert num_steps >= 1
    assert 0 < sigma_min <= sigma_max
    ramp = torch.linspace(
        0.0,
        1.0,
        num_steps,
        device=reference.device,
        dtype=reference.dtype,
    )
    min_inverse_rho = sigma_min ** (1.0 / rho)
    max_inverse_rho = sigma_max ** (1.0 / rho)
    positive = (
        max_inverse_rho + ramp * (min_inverse_rho - max_inverse_rho)
    ) ** rho
    return torch.cat([positive, reference.new_zeros(1)])


def sample_log_uniform_sigmas(reference, sigma_min, sigma_max, generator=None):
    uniform = reference.new_empty(reference.size(0))
    uniform.uniform_(generator=generator)
    return torch.exp(
        math.log(sigma_min)
        + uniform * (math.log(sigma_max) - math.log(sigma_min))
    )


def kabsch_aligned_rmsd(predicted_coords, true_coords, coord_mask):
    values = []
    for batch_index in range(predicted_coords.size(0)):
        mask = coord_mask[batch_index].bool()
        predicted = predicted_coords[batch_index, mask]
        target = true_coords[batch_index, mask]
        if predicted.size(0) == 0:
            values.append(predicted_coords.new_tensor(float('nan')))
            continue

        predicted = predicted - predicted.mean(dim=0, keepdim=True)
        target = target - target.mean(dim=0, keepdim=True)
        covariance = predicted.transpose(0, 1) @ target
        left_singular_vectors, _, right_singular_vectors_transposed = (
            torch.linalg.svd(covariance)
        )
        rotation = left_singular_vectors @ right_singular_vectors_transposed
        if torch.linalg.det(rotation) < 0:
            left_singular_vectors = left_singular_vectors.clone()
            left_singular_vectors[:, -1] = -left_singular_vectors[:, -1]
            rotation = (
                left_singular_vectors @ right_singular_vectors_transposed
            )
        aligned = predicted @ rotation
        squared_error = ((aligned - target) ** 2).sum(dim=-1)
        values.append(torch.sqrt(squared_error.mean()))
    return torch.stack(values)


def weighted_rigid_align(moving, target, atom_mask):
    batch = moving.size(0)
    moving_flat = moving.reshape(batch, -1, 3)
    target_flat = target.reshape(batch, -1, 3)
    mask_flat = atom_mask.reshape(batch, -1).to(moving.dtype)
    weights = mask_flat[..., None]
    denominator = weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
    moving_mean = (moving_flat * weights).sum(dim=1, keepdim=True) / denominator
    target_mean = (target_flat * weights).sum(dim=1, keepdim=True) / denominator
    moving_centered = moving_flat - moving_mean
    target_centered = target_flat - target_mean
    covariance = (target_centered * weights).transpose(1, 2) @ moving_centered
    left, _, right_t = torch.linalg.svd(covariance)
    det = torch.linalg.det(left @ right_t)
    correction = torch.ones(batch, 3, device=moving.device, dtype=moving.dtype)
    correction[:, -1] = det
    rotation = left @ torch.diag_embed(correction) @ right_t
    aligned = moving_centered @ rotation.transpose(-1, -2) + target_mean
    return aligned.reshape_as(moving)


def center_random_augmentation(coords, atom_mask, generator=None, second_coords=None):
    coords = masked_center(coords, atom_mask)
    if second_coords is not None:
        second_coords = masked_center(second_coords, atom_mask)
    rotation = random_rotation_matrix(coords, generator=generator)
    rotation_view = rotation.view(coords.size(0), *([1] * (coords.ndim - 2)), 3, 3)
    coords = torch.matmul(coords.unsqueeze(-2), rotation_view).squeeze(-2)
    translation = coords.new_empty(coords.size(0), 1, 1, 3)
    translation = translation.view(coords.size(0), *([1] * (coords.ndim - 2)), 3)
    translation.normal_(generator=generator)
    coords = (coords + translation) * atom_mask.to(coords.dtype)[..., None]
    if second_coords is not None:
        second_coords = torch.matmul(second_coords.unsqueeze(-2), rotation_view).squeeze(-2)
        second_coords = (second_coords + translation) * atom_mask.to(second_coords.dtype)[..., None]
    return coords, second_coords


@dataclass
class ProteinFoldingConfig:
    block_size: int = 64
    vocab_size: int = 64
    res_type_vocab_size: int = 33
    lm_dim: int = 64
    lm_layers: int = 2
    lm_heads: int = 4

    single_dim: int = 64
    pair_dim: int = 32
    atom_dim: int = 32
    atom_encoder_layers: int = 2
    atom_encoder_heads: int = 4
    atom_attention_window: int = 32
    diffusion_heads: int = 4
    pair_layers: int = 2
    recycle_loops: int = 2
    relative_position_bins: int = 16
    relative_chain_bins: int = 2

    time_dim: int = 32
    denoiser_layers: int = 2
    distance_rbf_bins: int = 16
    distogram_bins: int = 32
    plddt_bins: int = 50
    pae_bins: int = 32
    pde_bins: int = 32

    sigma_min: float = 0.1
    sigma_max: float = 20.0
    sigma_data: float = 10.0
    sampling_steps: int = 12
    confidence_rollout_steps: int = 4
    confidence_rollout_interval: int = 4

    mlm_mask_probability: float = 0.15
    dropout: float = 0.0

    def protein_lm_config(self):
        return ProteinLMConfig(
            vocab_size=self.vocab_size,
            context_size=self.block_size,
            embed_dim=self.lm_dim,
            lm_heads=self.lm_heads,
            lm_layers=self.lm_layers,
            mlm_mask_probability=self.mlm_mask_probability,
            dropout=self.dropout,
        )

    def inputs_dim(self):
        return self.atom_dim + 2 * self.res_type_vocab_size + 1


class AtomAttentionBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config.atom_encoder_heads
        self.head_dim = config.atom_dim // config.atom_encoder_heads
        assert config.atom_dim % config.atom_encoder_heads == 0
        assert self.head_dim % 2 == 0
        self.rotary_pairs = self.head_dim // 2
        self.window = config.atom_attention_window
        self.norm = nn.LayerNorm(config.atom_dim)
        self.qkv = nn.Linear(config.atom_dim, 3 * config.atom_dim)
        self.query_norm = nn.LayerNorm(self.head_dim)
        self.key_norm = nn.LayerNorm(self.head_dim)
        self.bond_bias = nn.Linear(1, config.atom_encoder_heads, bias=False)
        self.output = nn.Linear(config.atom_dim, config.atom_dim)
        self.transition_norm = nn.LayerNorm(config.atom_dim)
        self.transition = nn.Sequential(
            nn.Linear(config.atom_dim, 4 * config.atom_dim),
            nn.SiLU(),
            nn.Linear(4 * config.atom_dim, config.atom_dim),
        )
        frequencies = 1.0 / (
            10000 ** (torch.arange(self.rotary_pairs, dtype=torch.float32) / self.rotary_pairs)
        )
        self.register_buffer('rotary_frequencies', frequencies)

    def apply_atom_rope(self, tensor, ref_pos, ref_space_uid):
        batch, heads, atom_count, head_dim = tensor.shape
        coordinates = torch.cat(
            [
                ref_pos,
                ref_space_uid.to(ref_pos.dtype)[..., None],
            ],
            dim=-1,
        )
        coordinate_index = torch.arange(self.rotary_pairs, device=tensor.device) % 4
        rotary_source = coordinates[..., coordinate_index]
        angles = rotary_source * self.rotary_frequencies.to(ref_pos.dtype)
        cosine = torch.cos(angles).to(tensor.dtype)[:, None, :, :]
        sine = torch.sin(angles).to(tensor.dtype)[:, None, :, :]
        tensor_pairs = tensor.view(batch, heads, atom_count, self.rotary_pairs, 2)
        first = tensor_pairs[..., 0]
        second = tensor_pairs[..., 1]
        rotated = torch.stack(
            [
                first * cosine - second * sine,
                first * sine + second * cosine,
            ],
            dim=-1,
        )
        return rotated.view(batch, heads, atom_count, head_dim)

    def forward(self, atoms, atom_mask, ref_pos, ref_space_uid, atom_bonds):
        batch, atom_count, width = atoms.shape
        normalized = self.norm(atoms)
        query, key, value = self.qkv(normalized).chunk(3, dim=-1)
        query = query.view(batch, atom_count, self.heads, self.head_dim).transpose(1, 2)
        key = key.view(batch, atom_count, self.heads, self.head_dim).transpose(1, 2)
        value = value.view(batch, atom_count, self.heads, self.head_dim).transpose(1, 2)
        query = self.apply_atom_rope(self.query_norm(query), ref_pos, ref_space_uid)
        key = self.apply_atom_rope(self.key_norm(key), ref_pos, ref_space_uid)
        scores = query @ key.transpose(-2, -1)
        scores = scores / math.sqrt(self.head_dim)
        bond_bias = self.bond_bias(atom_bonds.to(atoms.dtype)[..., None])
        scores = scores + bond_bias.permute(0, 3, 1, 2)

        atom_positions = torch.arange(atom_count, device=atoms.device)
        local_window = (atom_positions[:, None] - atom_positions[None, :]).abs() <= self.window
        same_reference_space = ref_space_uid[:, :, None] == ref_space_uid[:, None, :]
        key_mask = atom_mask[:, None, None, :]
        query_mask = atom_mask[:, None, :, None]
        bonded_atoms = atom_bonds.bool()
        attention_mask = (
            (local_window[None, None] | bonded_atoms[:, None])
            & (same_reference_space[:, None] | bonded_atoms[:, None])
            & key_mask
        )
        scores = scores.masked_fill(~attention_mask, torch.finfo(scores.dtype).min)
        scores = torch.where(query_mask, scores, torch.zeros_like(scores))
        weights = F.softmax(scores, dim=-1)
        weights = weights * query_mask.to(weights.dtype)
        attended = weights @ value
        attended = attended.transpose(1, 2).contiguous().view(batch, atom_count, width)
        atoms = atoms + self.output(attended)
        atoms = atoms + self.transition(self.transition_norm(atoms))
        return atoms * atom_mask.to(atoms.dtype)[..., None]


class AtomInputEncoder(nn.Module):
    def __init__(self, config, structure_prediction=False, output_dim=None):
        super().__init__()
        self.structure_prediction = structure_prediction
        self.reference_projection = nn.Linear(3, config.atom_dim)
        self.element_embedding = nn.Embedding(
            ATOM_ELEMENT_VOCAB_SIZE,
            config.atom_dim,
            padding_idx=0,
        )
        self.charge_projection = nn.Linear(1, config.atom_dim)
        self.name_char_embedding = nn.Embedding(
            ATOM_NAME_CHAR_VOCAB_SIZE,
            config.atom_dim,
            padding_idx=0,
        )
        self.mask_projection = nn.Linear(1, config.atom_dim)
        self.space_uid_embedding = nn.Embedding(
            config.block_size + 1,
            config.atom_dim,
            padding_idx=0,
        )
        if structure_prediction:
            self.coords_projection = nn.Linear(6, config.atom_dim, bias=False)
        self.blocks = nn.ModuleList(
            [AtomAttentionBlock(config) for _ in range(config.atom_encoder_layers)]
        )
        output_dim = config.single_dim if output_dim is None else output_dim
        self.atom_to_token = nn.Linear(config.atom_dim, output_dim, bias=False)

    def forward(
        self,
        ref_pos,
        atom_attention_mask,
        ref_space_uid,
        atom_element,
        atom_charge,
        atom_name_chars,
        atom_to_token,
        atom_bonds,
        token_count=None,
        r_l=None,
        pred_r1=None,
        return_intermediates=False,
    ):
        batch, atom_count, _ = ref_pos.shape
        atom_attention_mask = atom_attention_mask.bool()
        validate_ref_space_uid(
            ref_space_uid,
            atom_attention_mask,
            self.space_uid_embedding.num_embeddings - 1,
        )
        name_features = self.name_char_embedding(
            atom_name_chars.clamp_min(0).clamp_max(ATOM_NAME_CHAR_VOCAB_SIZE - 1)
        ).sum(dim=-2)
        features = (
            self.reference_projection(ref_pos)
            + self.element_embedding(atom_element.clamp_min(0).clamp_max(ATOM_ELEMENT_VOCAB_SIZE - 1))
            + self.charge_projection(atom_charge.to(ref_pos.dtype)[..., None])
            + name_features
            + self.mask_projection(atom_attention_mask.to(ref_pos.dtype)[..., None])
            + self.space_uid_embedding(ref_space_uid.clamp_min(0).clamp_max(self.space_uid_embedding.num_embeddings - 1))
        )
        if self.structure_prediction and r_l is not None:
            if pred_r1 is None:
                pred_r1 = torch.zeros_like(r_l)
            relative_coords = r_l - pred_r1
            r_l_squared = (r_l * r_l).sum(dim=-1, keepdim=True)
            pred_r1_squared = (pred_r1 * pred_r1).sum(dim=-1, keepdim=True)
            relative_squared = (relative_coords * relative_coords).sum(dim=-1, keepdim=True)
            coord_features = torch.cat(
                [
                    torch.sqrt(r_l_squared + 1e-8),
                    torch.sqrt(pred_r1_squared + 1e-8),
                    torch.sqrt(relative_squared + 1e-8),
                    (r_l * pred_r1).sum(dim=-1, keepdim=True),
                    r_l_squared,
                    pred_r1_squared,
                ],
                dim=-1,
            )
            features = features + self.coords_projection(coord_features)
        atom_states = features * atom_attention_mask.to(features.dtype)[..., None]
        atom_trajectory = []
        for block in self.blocks:
            atom_states = block(atom_states, atom_attention_mask, ref_pos, ref_space_uid, atom_bonds)
            if return_intermediates:
                atom_trajectory.append(atom_states)
        if token_count is None:
            token_count = int(atom_to_token.clamp_min(0).max().item()) + 1
            token_count = max(token_count, 1)
        token_features = scatter_atom_to_token(
            F.relu(self.atom_to_token(atom_states)),
            atom_to_token,
            token_count,
            atom_attention_mask,
        )
        return {
            'token_features': token_features,
            'atom_features': atom_states,
            'atom_base_features': features,
            'atom_trajectory': atom_trajectory,
        }


class InputsEmbedder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.atom_input_encoder = AtomInputEncoder(
            config,
            structure_prediction=False,
            output_dim=config.atom_dim,
        )
        self.output_norm = nn.LayerNorm(config.inputs_dim())

    def forward(
        self,
        res_type,
        token_attention_mask,
        ref_pos,
        atom_attention_mask,
        ref_space_uid,
        atom_element,
        atom_charge,
        atom_name_chars,
        atom_to_token,
        atom_bonds,
        profile=None,
        deletion_mean=None,
        return_intermediates=False,
    ):
        atom_output = self.atom_input_encoder(
            ref_pos,
            atom_attention_mask,
            ref_space_uid,
            atom_element,
            atom_charge,
            atom_name_chars,
            atom_to_token,
            atom_bonds,
            token_count=res_type.size(1),
            return_intermediates=return_intermediates,
        )
        aatype = F.one_hot(
            res_type.clamp_min(0).clamp_max(self.config.res_type_vocab_size - 1),
            num_classes=self.config.res_type_vocab_size,
        ).to(atom_output['token_features'].dtype)
        aatype = aatype * token_attention_mask.to(aatype.dtype)[..., None]
        if profile is None:
            profile = aatype
        if deletion_mean is None:
            deletion_mean = torch.zeros_like(token_attention_mask, dtype=aatype.dtype)
        x_inputs = torch.cat(
            [
                atom_output['token_features'],
                aatype,
                profile.to(aatype.dtype),
                deletion_mean.to(aatype.dtype)[..., None],
            ],
            dim=-1,
        )
        x_inputs = self.output_norm(x_inputs)
        x_inputs = x_inputs * token_attention_mask.to(x_inputs.dtype)[..., None]
        return {
            'x_inputs': x_inputs,
            'atom_features': atom_output['atom_features'],
            'atom_token_features': atom_output['token_features'],
            'aatype': aatype,
            'profile': profile,
            'deletion_mean': deletion_mean,
            'atom_encoder': atom_output,
        }


class SingleToPair(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.downproject = nn.Linear(input_dim, output_dim)
        self.output = nn.Sequential(
            nn.Linear(2 * output_dim, output_dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, single):
        projected = self.downproject(single)
        product = projected[:, :, None, :] * projected[:, None, :, :]
        difference = projected[:, :, None, :] - projected[:, None, :, :]
        return self.output(torch.cat([product, difference], dim=-1))


class LanguageModelShim(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_weights = nn.Parameter(torch.zeros(config.lm_layers + 1))
        self.projection = nn.Sequential(
            nn.LayerNorm(config.lm_dim),
            nn.Linear(config.lm_dim, config.pair_dim, bias=False),
        )
        self.single_to_pair = SingleToPair(config.pair_dim, config.pair_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden_states, residue_mask, lm_dropout=False):
        assert hidden_states.ndim == 4
        weights = F.softmax(self.layer_weights, dim=0)
        projected_layers = self.projection(hidden_states)
        mixed = (projected_layers * weights[None, None, :, None]).sum(dim=2)
        mixed = mixed * residue_mask.to(mixed.dtype)[..., None]
        pair = self.single_to_pair(mixed)
        if lm_dropout:
            pair = self.dropout(pair)
        return pair * sequence_pair_mask(residue_mask).to(pair.dtype)[..., None]


class RelativePositionEncoding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        residue_features = 2 * config.relative_position_bins + 2
        token_features = 2 * config.relative_position_bins + 2
        chain_features = 2 * config.relative_chain_bins + 2
        self.output = nn.Linear(
            residue_features + token_features + chain_features + 1,
            config.pair_dim,
            bias=False,
        )

    def forward(self, residue_index, token_index, asym_id, sym_id, entity_id):
        same_chain = asym_id[:, :, None] == asym_id[:, None, :]
        same_residue = residue_index[:, :, None] == residue_index[:, None, :]
        same_entity = entity_id[:, :, None] == entity_id[:, None, :]

        residue_delta = residue_index[:, :, None] - residue_index[:, None, :]
        residue_delta = torch.clamp(
            residue_delta + self.config.relative_position_bins,
            0,
            2 * self.config.relative_position_bins,
        )
        residue_delta = torch.where(
            same_chain,
            residue_delta,
            torch.full_like(residue_delta, 2 * self.config.relative_position_bins + 1),
        )
        residue_one_hot = F.one_hot(
            residue_delta,
            2 * self.config.relative_position_bins + 2,
        )

        token_delta = token_index[:, :, None] - token_index[:, None, :]
        token_delta = torch.clamp(
            token_delta + self.config.relative_position_bins,
            0,
            2 * self.config.relative_position_bins,
        )
        token_delta = torch.where(
            same_chain & same_residue,
            token_delta,
            torch.full_like(token_delta, 2 * self.config.relative_position_bins + 1),
        )
        token_one_hot = F.one_hot(
            token_delta,
            2 * self.config.relative_position_bins + 2,
        )

        chain_delta = sym_id[:, :, None] - sym_id[:, None, :]
        chain_delta = torch.clamp(
            chain_delta + self.config.relative_chain_bins,
            0,
            2 * self.config.relative_chain_bins,
        )
        chain_delta = torch.where(
            same_chain,
            torch.full_like(chain_delta, 2 * self.config.relative_chain_bins + 1),
            chain_delta,
        )
        chain_one_hot = F.one_hot(chain_delta, 2 * self.config.relative_chain_bins + 2)
        features = torch.cat(
            [
                residue_one_hot,
                token_one_hot,
                same_entity[..., None],
                chain_one_hot,
            ],
            dim=-1,
        ).to(torch.float32)
        return self.output(features)


class PairEmbedder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.z_init_1 = nn.Linear(config.inputs_dim(), config.pair_dim, bias=False)
        self.z_init_2 = nn.Linear(config.inputs_dim(), config.pair_dim, bias=False)
        self.relative_encoding = RelativePositionEncoding(config)
        self.token_bond_projection = nn.Linear(1, config.pair_dim, bias=False)

    def forward(
        self,
        x_inputs,
        token_attention_mask,
        token_bonds,
        residue_index,
        token_index,
        asym_id,
        sym_id,
        entity_id,
        return_intermediates=False,
    ):
        z_init = (
            self.z_init_1(x_inputs).unsqueeze(2)
            + self.z_init_2(x_inputs).unsqueeze(1)
        )
        relative_position_features = self.relative_encoding(
            residue_index,
            token_index,
            asym_id,
            sym_id,
            entity_id,
        )
        if token_bonds.ndim == 3:
            token_bonds = token_bonds[..., None]
        token_bond_features = self.token_bond_projection(token_bonds.to(x_inputs.dtype))

        pair_mask = sequence_pair_mask(token_attention_mask)
        pair = z_init + relative_position_features + token_bond_features
        pair = pair * pair_mask.to(pair.dtype)[..., None]

        intermediates = None
        if return_intermediates:
            intermediates = {
                'x_inputs': x_inputs,
                'z_init': z_init,
                'token_bonds': token_bonds,
            }
        return pair, relative_position_features, token_bond_features, intermediates


class TriangleMultiplicativeUpdate(nn.Module):
    def __init__(self, config, flow):
        super().__init__()
        assert flow in {'outgoing', 'incoming'}
        self.flow = flow
        self.norm = nn.LayerNorm(config.pair_dim)
        self.value_a = nn.Linear(config.pair_dim, config.pair_dim)
        self.value_b = nn.Linear(config.pair_dim, config.pair_dim)
        self.gate_a = nn.Linear(config.pair_dim, config.pair_dim)
        self.gate_b = nn.Linear(config.pair_dim, config.pair_dim)
        self.output_gate = nn.Linear(config.pair_dim, config.pair_dim)
        self.output = nn.Linear(config.pair_dim, config.pair_dim)

    def forward(self, pair, pair_mask, return_intermediates=False):
        normalized = self.norm(pair)
        pair_values = pair_mask.to(pair.dtype)[..., None]
        first_values = (
            torch.sigmoid(self.gate_a(normalized)) * self.value_a(normalized)
        )
        second_values = (
            torch.sigmoid(self.gate_b(normalized)) * self.value_b(normalized)
        )
        first_values = first_values * pair_values
        second_values = second_values * pair_values

        first_values_by_channel = first_values.permute(0, 3, 1, 2)
        second_values_by_channel = second_values.permute(0, 3, 1, 2)
        if self.flow == 'outgoing':
            # For pair (i, j), sum paths through shared residue k:
            # first[i, k] * second[j, k].
            triangle_features = (
                first_values_by_channel
                @ second_values_by_channel.transpose(-2, -1)
            )
        else:
            # For pair (i, j), sum paths through shared residue k:
            # first[k, i] * second[k, j].
            triangle_features = (
                first_values_by_channel.transpose(-2, -1)
                @ second_values_by_channel
            )
        triangle_features = triangle_features.permute(0, 2, 3, 1)

        shared_residue_count = (
            pair_mask.any(dim=2).sum(dim=1).to(pair.dtype).clamp_min(1.0)
        )
        triangle_features = triangle_features / torch.sqrt(
            shared_residue_count
        )[:, None, None, None]
        output_gate = torch.sigmoid(self.output_gate(normalized))
        update = (
            self.output(triangle_features) * output_gate * pair_values
        )

        intermediates = None
        if return_intermediates:
            intermediates = {
                'normalized': normalized,
                'first_values': first_values,
                'second_values': second_values,
                'triangle_features': triangle_features,
                'output_gate': output_gate,
            }
        return update, intermediates


class PairTransition(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm = nn.LayerNorm(config.pair_dim)
        self.fc = nn.Linear(config.pair_dim, 8 * config.pair_dim)
        self.output = nn.Linear(4 * config.pair_dim, config.pair_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, pair, pair_mask):
        gate, values = self.fc(self.norm(pair)).chunk(2, dim=-1)
        update = self.output(F.silu(gate) * values)
        update = self.dropout(update)
        return update * pair_mask.to(update.dtype)[..., None]


class PairBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.triangle_outgoing = TriangleMultiplicativeUpdate(config, 'outgoing')
        self.triangle_incoming = TriangleMultiplicativeUpdate(config, 'incoming')
        self.transition = PairTransition(config)

    def forward(self, pair, pair_mask, return_intermediates=False):
        outgoing_update, outgoing_intermediates = self.triangle_outgoing(
            pair,
            pair_mask,
            return_intermediates=return_intermediates,
        )
        pair = pair + outgoing_update
        incoming_update, incoming_intermediates = self.triangle_incoming(
            pair,
            pair_mask,
            return_intermediates=return_intermediates,
        )
        pair = pair + incoming_update
        transition_update = self.transition(pair, pair_mask)
        pair = pair + transition_update
        pair = pair * pair_mask.to(pair.dtype)[..., None]

        intermediates = None
        if return_intermediates:
            intermediates = {
                'triangle_outgoing': outgoing_intermediates,
                'triangle_incoming': incoming_intermediates,
                'outgoing_update': outgoing_update,
                'incoming_update': incoming_update,
                'transition_update': transition_update,
            }
        return pair, intermediates


class RecyclingTrunk(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.recycle_loops = config.recycle_loops
        self.input_norm = nn.LayerNorm(config.pair_dim)
        self.parcae_log_a = nn.Parameter(torch.zeros(config.pair_dim))
        decay = math.sqrt(1.0 / 5.0)
        delta = -math.log(decay)
        self.parcae_log_delta = nn.Parameter(
            torch.full((config.pair_dim,), math.log(math.exp(delta) - 1.0))
        )
        self.parcae_b_cont = nn.Parameter(torch.eye(config.pair_dim))
        self.blocks = nn.ModuleList(
            [PairBlock(config) for _ in range(config.pair_layers)]
        )
        self.readout = nn.Linear(config.pair_dim, config.pair_dim, bias=False)
        nn.init.eye_(self.readout.weight)
        self.coda = nn.ModuleList(
            [PairBlock(config) for _ in range(max(1, config.pair_layers // 2))]
        )

    def forward(self, pair_initial, lm_pair, pair_mask, return_intermediates=False, generator=None):
        std = math.sqrt(2.0 / (5.0 * pair_initial.size(-1)))
        pair = torch.empty_like(pair_initial)
        nn.init.trunc_normal_(
            pair,
            mean=0.0,
            std=std,
            a=-3 * std,
            b=3 * std,
            generator=generator,
        )
        recycle_states = []
        block_details = []
        mask = pair_mask.to(pair.dtype)[..., None]
        delta = F.softplus(self.parcae_log_delta)
        a = torch.exp(-delta * torch.exp(self.parcae_log_a))
        b_matrix = delta[:, None] * self.parcae_b_cont
        a = a.view(1, 1, 1, -1).to(pair)
        b_matrix = b_matrix.to(pair)

        for _ in range(self.recycle_loops + 1):
            z_inject_pair = pair_initial
            if lm_pair is not None:
                z_inject_pair = z_inject_pair + lm_pair.to(z_inject_pair.dtype)
            injected_pair = self.input_norm(z_inject_pair)
            pair = (a * pair + F.linear(injected_pair.to(pair.dtype), b_matrix)) * mask
            current_details = []
            for block in self.blocks:
                pair, details = block(
                    pair,
                    pair_mask,
                    return_intermediates=return_intermediates,
                )
                if return_intermediates:
                    current_details.append(details)
            if return_intermediates:
                recycle_states.append(pair)
                block_details.append(current_details)

        pair = self.readout(pair) * mask
        coda_details = []
        for block in self.coda:
            pair, details = block(
                pair,
                pair_mask,
                return_intermediates=return_intermediates,
            )
            if return_intermediates:
                coda_details.append(details)
        return (
            pair,
            recycle_states if return_intermediates else None,
            {
                'recurrent_blocks': block_details,
                'coda_blocks': coda_details,
                'a': a,
                'b_matrix': b_matrix,
                'lm_pair_injected': lm_pair is not None,
            } if return_intermediates else None,
        )


class DistogramHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm = nn.LayerNorm(config.pair_dim)
        self.output = nn.Linear(config.pair_dim, config.distogram_bins)
        boundaries = torch.linspace(2.0, 22.0, config.distogram_bins - 1)
        self.register_buffer('boundaries', boundaries)

    def forward(self, pair, pair_mask):
        symmetric_pair = pair + pair.transpose(1, 2)
        logits = self.output(self.norm(symmetric_pair))
        logits = 0.5 * (logits + logits.transpose(1, 2))
        return logits * pair_mask.to(logits.dtype)[..., None]

class SinusoidalNoiseEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.time_dim = config.time_dim
        feature_dim = max(1, config.time_dim // 2)
        frequencies = torch.exp(
            torch.linspace(math.log(1.0), math.log(1000.0), feature_dim)
        )
        self.register_buffer('frequencies', frequencies)
        self.mlp = nn.Sequential(
            nn.Linear(2 * feature_dim, config.time_dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(config.time_dim, config.time_dim),
        )

    def forward(self, sigma):
        log_sigma = torch.log(sigma.clamp_min(1e-8))
        angles = log_sigma[:, None] * self.frequencies[None, :]
        features = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return self.mlp(features)


class RadialBasisEmbedding(nn.Module):
    def __init__(self, num_bins, minimum=0.0, maximum=30.0):
        super().__init__()
        centers = torch.linspace(minimum, maximum, num_bins)
        spacing = (maximum - minimum) / max(1, num_bins - 1)
        self.register_buffer('centers', centers)
        self.gamma = 1.0 / max(spacing * spacing, 1e-6)

    def forward(self, distances):
        difference = distances[..., None] - self.centers
        return torch.exp(-self.gamma * difference * difference)


class AdaptiveLayerNorm(nn.Module):
    def __init__(self, width, condition_width):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.scale = nn.Linear(condition_width, width)
        self.shift = nn.Linear(condition_width, width)

    def forward(self, values, condition):
        scale = 1.0 + self.scale(condition)
        shift = self.shift(condition)
        return self.norm(values) * scale + shift


class DiffusionConditioning(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.noise_embedding = SinusoidalNoiseEmbedding(config)
        self.single_norm = nn.LayerNorm(config.inputs_dim())
        self.single_projection = nn.Linear(config.inputs_dim(), config.single_dim, bias=False)
        self.noise_norm = nn.LayerNorm(config.time_dim)
        self.noise_projection = nn.Linear(config.time_dim, config.single_dim, bias=False)
        self.single_transitions = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(config.single_dim),
                    nn.Linear(config.single_dim, 2 * config.single_dim),
                    nn.SiLU(),
                    nn.Linear(2 * config.single_dim, config.single_dim),
                )
                for _ in range(2)
            ]
        )
        self.pair_norm = nn.LayerNorm(2 * config.pair_dim)
        self.pair_projection = nn.Linear(2 * config.pair_dim, config.pair_dim, bias=False)
        self.pair_transitions = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(config.pair_dim),
                    nn.Linear(config.pair_dim, 2 * config.pair_dim),
                    nn.SiLU(),
                    nn.Linear(2 * config.pair_dim, config.pair_dim),
                )
                for _ in range(2)
            ]
        )

    def forward(self, sigma, x_inputs, pair, relative_position_encoding, residue_mask):
        noise = self.noise_projection(self.noise_norm(self.noise_embedding(sigma)))
        conditioned_single = self.single_projection(self.single_norm(x_inputs))
        conditioned_single = conditioned_single + noise[:, None, :]
        for transition in self.single_transitions:
            conditioned_single = conditioned_single + transition(conditioned_single)
        conditioned_single = conditioned_single * residue_mask.to(conditioned_single.dtype)[..., None]

        conditioned_pair = torch.cat([pair, relative_position_encoding], dim=-1)
        conditioned_pair = self.pair_projection(self.pair_norm(conditioned_pair))
        for transition in self.pair_transitions:
            conditioned_pair = conditioned_pair + transition(conditioned_pair)
        conditioned_pair = conditioned_pair * sequence_pair_mask(residue_mask).to(
            conditioned_pair.dtype
        )[..., None]
        return conditioned_single, conditioned_pair


class PairBiasedTokenAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config.diffusion_heads
        self.head_dim = config.single_dim // config.diffusion_heads
        assert config.single_dim % config.diffusion_heads == 0
        self.adaln = AdaptiveLayerNorm(config.single_dim, config.single_dim)
        self.q = nn.Linear(config.single_dim, config.single_dim)
        self.kv = nn.Linear(config.single_dim, 2 * config.single_dim, bias=False)
        self.gate = nn.Linear(config.single_dim, config.single_dim, bias=False)
        self.output = nn.Linear(config.single_dim, config.single_dim, bias=False)
        self.pair_norm = nn.LayerNorm(config.pair_dim)
        self.pair_bias = nn.Linear(config.pair_dim, config.diffusion_heads, bias=False)
        self.output_gate = nn.Linear(config.single_dim, config.single_dim)

    def forward(self, tokens, condition, pair, residue_mask):
        batch, length, _ = tokens.shape
        normalized = self.adaln(tokens, condition)
        query = self.q(normalized).view(batch, length, self.heads, self.head_dim)
        key, value = self.kv(normalized).chunk(2, dim=-1)
        key = key.view(batch, length, self.heads, self.head_dim)
        value = value.view(batch, length, self.heads, self.head_dim)
        scores = torch.matmul(
            query.permute(0, 2, 1, 3),
            key.permute(0, 2, 3, 1),
        )
        scores = scores / math.sqrt(self.head_dim)
        scores = scores + self.pair_bias(self.pair_norm(pair)).permute(0, 3, 1, 2)
        scores = scores.masked_fill(
            ~residue_mask[:, None, None, :],
            torch.finfo(scores.dtype).min,
        )
        weights = F.softmax(scores, dim=-1)
        attended = weights @ value.permute(0, 2, 1, 3)
        attended = attended.permute(0, 2, 1, 3).contiguous()
        attended = attended.view(batch, length, self.heads * self.head_dim)
        gated = torch.sigmoid(self.gate(normalized)) * attended
        update = self.output(gated)
        update = torch.sigmoid(self.output_gate(condition)) * update
        return update * residue_mask.to(update.dtype)[..., None]


class ConditionedTransition(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.adaln = AdaptiveLayerNorm(config.single_dim, config.single_dim)
        self.projection = nn.Linear(config.single_dim, 4 * config.single_dim)
        self.output = nn.Linear(2 * config.single_dim, config.single_dim, bias=False)
        self.output_gate = nn.Linear(config.single_dim, config.single_dim)

    def forward(self, tokens, condition, residue_mask):
        normalized = self.adaln(tokens, condition)
        gate, values = self.projection(normalized).chunk(2, dim=-1)
        update = self.output(F.silu(gate) * values)
        update = torch.sigmoid(self.output_gate(condition)) * update
        return update * residue_mask.to(update.dtype)[..., None]


class DiffusionTransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = PairBiasedTokenAttention(config)
        self.transition = ConditionedTransition(config)

    def forward(self, tokens, condition, pair, residue_mask):
        tokens = tokens + self.attention(tokens, condition, pair, residue_mask)
        tokens = tokens + self.transition(tokens, condition, residue_mask)
        return tokens * residue_mask.to(tokens.dtype)[..., None]


class StructureDenoiser(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.conditioning = DiffusionConditioning(config)
        self.noisy_atom_encoder = AtomInputEncoder(
            config,
            structure_prediction=True,
            output_dim=config.single_dim,
        )
        self.token_projection = nn.Linear(config.single_dim, config.single_dim)
        self.transformer = nn.ModuleList(
            [DiffusionTransformerBlock(config) for _ in range(config.denoiser_layers)]
        )
        self.token_norm = nn.LayerNorm(config.single_dim)
        self.token_to_atom = nn.Linear(config.single_dim, config.atom_dim, bias=False)
        self.atom_decoder = nn.ModuleList(
            [AtomAttentionBlock(config) for _ in range(config.atom_encoder_layers)]
        )
        self.atom_output_norm = nn.LayerNorm(config.atom_dim)
        self.atom_output = nn.Linear(config.atom_dim, 2, bias=False)

    def edm_scaling(self, sigma):
        sigma_data = self.config.sigma_data
        denominator = torch.sqrt(sigma * sigma + sigma_data * sigma_data)
        input_scale = 1.0 / denominator
        skip_scale = (sigma_data * sigma_data) / (
            sigma * sigma + sigma_data * sigma_data
        )
        output_scale = sigma * sigma_data / denominator
        return input_scale, skip_scale, output_scale

    def forward(
        self,
        noisy_atom_coords,
        x_inputs,
        pair,
        relative_position_encoding,
        ref_pos,
        atom_attention_mask,
        ref_space_uid,
        atom_element,
        atom_charge,
        atom_name_chars,
        atom_to_token,
        atom_bonds,
        sigma,
        token_attention_mask,
        atom_mask,
        return_intermediates=False,
    ):
        input_scale, skip_scale, output_scale = self.edm_scaling(sigma)
        noisy_scaled = input_scale[:, None, None] * noisy_atom_coords
        noisy_scaled = masked_center(noisy_scaled, atom_mask)
        conditioned_single, conditioned_pair = self.conditioning(
            sigma,
            x_inputs,
            pair,
            relative_position_encoding,
            token_attention_mask,
        )
        noisy_atom_output = self.noisy_atom_encoder(
            ref_pos,
            atom_attention_mask,
            ref_space_uid,
            atom_element,
            atom_charge,
            atom_name_chars,
            atom_to_token,
            atom_bonds,
            token_count=token_attention_mask.size(1),
            r_l=noisy_scaled,
            return_intermediates=return_intermediates,
        )
        noisy_atom_features = noisy_atom_output['atom_features']
        noisy_token_features = noisy_atom_output['token_features']
        nodes = self.token_projection(conditioned_single) + noisy_token_features
        nodes = nodes * token_attention_mask.to(nodes.dtype)[..., None]

        node_trajectory = []
        for block in self.transformer:
            nodes = block(nodes, conditioned_single, conditioned_pair, token_attention_mask)
            if return_intermediates:
                node_trajectory.append(nodes)

        nodes = self.token_norm(nodes)
        atom_state = noisy_atom_features + gather_token_to_atom(
            self.token_to_atom(nodes),
            atom_to_token,
        )
        for block in self.atom_decoder:
            atom_state = block(atom_state, atom_mask, ref_pos, ref_space_uid, atom_bonds)
        coordinate_weights = self.atom_output(self.atom_output_norm(atom_state))
        token_coord_center = scatter_atom_to_token(
            noisy_scaled,
            atom_to_token,
            token_attention_mask.size(1),
            atom_mask,
        )
        relative_to_token = noisy_scaled - gather_token_to_atom(
            token_coord_center,
            atom_to_token,
        )
        coordinate_update = (
            coordinate_weights[..., 0:1] * noisy_scaled
            + coordinate_weights[..., 1:2] * relative_to_token
        )
        coordinate_update = coordinate_update * atom_mask.to(coordinate_update.dtype)[..., None]

        pred_atom_coords = (
            skip_scale[:, None, None] * noisy_atom_coords
            + output_scale[:, None, None] * coordinate_update
        )
        pred_atom_coords = masked_center(pred_atom_coords, atom_mask)

        output = {
            'pred_atom_coords': pred_atom_coords,
            'nodes': nodes,
        }
        if return_intermediates:
            output['intermediates'] = {
                'input_scale': input_scale,
                'skip_scale': skip_scale,
                'output_scale': output_scale,
                'conditioned_single': conditioned_single,
                'conditioned_pair': conditioned_pair,
                'noisy_atom_features': noisy_atom_features,
                'noisy_token_features': noisy_token_features,
                'coordinate_update': coordinate_update,
                'coordinate_weights': coordinate_weights,
                'node_trajectory': node_trajectory,
                'atom_encoder': noisy_atom_output,
            }
        return output


class RowAttentionPooling(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = nn.Linear(config.pair_dim, 1, bias=False)
        self.output = nn.Linear(config.pair_dim, config.single_dim, bias=False)

    def forward(self, pair, residue_mask):
        scores = self.attention(pair).squeeze(-1)
        scores = scores.masked_fill(
            ~residue_mask[:, None, :],
            torch.finfo(scores.dtype).min,
        )
        weights = F.softmax(scores, dim=-1)
        pooled = (weights[..., None] * pair).sum(dim=2)
        return self.output(pooled) * residue_mask.to(pair.dtype)[..., None]


class ConfidenceHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.distance_boundaries = torch.linspace(2.0, 22.0, config.distogram_bins - 1)
        self.register_buffer('confidence_dist_boundaries', self.distance_boundaries)
        self.distance_bin_embedding = nn.Embedding(config.distogram_bins, config.pair_dim)
        self.inputs_norm = nn.LayerNorm(config.inputs_dim())
        self.inputs_to_single = nn.Linear(config.inputs_dim(), config.single_dim, bias=False)
        self.single_to_pair_a = nn.Linear(config.inputs_dim(), config.pair_dim, bias=False)
        self.single_to_pair_b = nn.Linear(config.inputs_dim(), config.pair_dim, bias=False)
        self.single_to_pair_prod_a = nn.Linear(config.inputs_dim(), config.pair_dim, bias=False)
        self.single_to_pair_prod_b = nn.Linear(config.inputs_dim(), config.pair_dim, bias=False)
        self.single_to_pair_prod_out = nn.Linear(config.pair_dim, config.pair_dim, bias=False)
        self.pair_norm = nn.LayerNorm(config.pair_dim)
        self.trunk = nn.ModuleList(
            [PairBlock(config) for _ in range(max(1, config.pair_layers // 2))]
        )
        self.row_pooling = RowAttentionPooling(config)
        self.single_norm = nn.LayerNorm(config.single_dim)
        self.scalar_plddt = nn.Linear(config.single_dim, 1)
        self.plddt_norm = nn.LayerNorm(config.single_dim)
        self.plddt_weight = nn.Parameter(
            torch.zeros(len(ATOM_SLOTS), config.single_dim, config.plddt_bins)
        )
        nn.init.normal_(self.plddt_weight, std=0.02)
        self.pae_head = nn.Linear(config.pair_dim, config.pae_bins)
        self.pde_head = nn.Linear(config.pair_dim, config.pde_bins)
        self.resolved_weight = nn.Parameter(
            torch.zeros(MAX_ATOMS_PER_TOKEN, config.single_dim, 2)
        )
        nn.init.normal_(self.resolved_weight, std=0.02)
        plddt_centers = torch.linspace(0.0, 1.0, config.plddt_bins)
        error_centers = torch.linspace(0.0, 32.0, config.pae_bins)
        distance_error_centers = torch.linspace(0.0, 32.0, config.pde_bins)
        self.register_buffer('plddt_centers', plddt_centers)
        self.register_buffer('pae_error_centers', error_centers)
        self.register_buffer('pde_error_centers', distance_error_centers)

    def forward(
        self,
        x_inputs,
        pair,
        sampled_atom_coords,
        distogram_atom_idx,
        token_attention_mask,
        atom_to_token,
        atom_mask,
        asym_id,
        mol_type,
        relative_position_encoding=None,
        token_bond_features=None,
    ):
        pair_mask = sequence_pair_mask(token_attention_mask)
        sampled_coords = gather_rep_atom_coords(
            sampled_atom_coords,
            distogram_atom_idx,
        )
        distances = pairwise_distances(sampled_coords)
        distance_bins = torch.bucketize(
            distances.contiguous(),
            self.confidence_dist_boundaries,
        )
        confidence_pair = self.pair_norm(pair)
        if relative_position_encoding is not None:
            confidence_pair = confidence_pair + relative_position_encoding
        if token_bond_features is not None:
            confidence_pair = confidence_pair + token_bond_features
        confidence_pair = confidence_pair + self.distance_bin_embedding(distance_bins)
        normalized_inputs = self.inputs_norm(x_inputs)
        confidence_pair = confidence_pair + self.single_to_pair_a(normalized_inputs).unsqueeze(2)
        confidence_pair = confidence_pair + self.single_to_pair_b(normalized_inputs).unsqueeze(1)
        confidence_pair = confidence_pair + self.single_to_pair_prod_out(
            self.single_to_pair_prod_a(normalized_inputs)[:, :, None, :]
            * self.single_to_pair_prod_b(normalized_inputs)[:, None, :, :]
        )
        confidence_pair = confidence_pair * pair_mask.to(confidence_pair.dtype)[..., None]
        for block in self.trunk:
            confidence_pair, _ = block(confidence_pair, pair_mask, return_intermediates=False)

        residue_features = self.inputs_to_single(normalized_inputs)
        residue_features = residue_features + self.row_pooling(confidence_pair, token_attention_mask)
        residue_features = residue_features * token_attention_mask.to(residue_features.dtype)[..., None]
        logits = self.scalar_plddt(residue_features).squeeze(-1)
        logits = logits * token_attention_mask.to(logits.dtype)

        atom_residue_features = gather_token_to_atom(residue_features, atom_to_token)
        atom_indices = compute_intra_token_index(atom_to_token, atom_mask)
        plddt_weights = self.plddt_weight[atom_indices]
        resolved_weights = self.resolved_weight[atom_indices]
        plddt_logits = (
            self.plddt_norm(atom_residue_features).unsqueeze(-2)
            @ plddt_weights
        ).squeeze(-2)
        plddt_probabilities = F.softmax(plddt_logits, dim=-1)
        plddt_probabilities = plddt_probabilities * atom_mask.to(
            plddt_probabilities.dtype
        )[..., None]
        plddt_per_atom = 100.0 * (
            plddt_probabilities * self.plddt_centers
        ).sum(dim=-1)
        plddt_per_atom = plddt_per_atom * atom_mask.to(plddt_per_atom.dtype)
        plddt_logits = plddt_logits * atom_mask.to(plddt_logits.dtype)[..., None]
        atom_values = atom_mask.to(plddt_per_atom.dtype)
        plddt_sum = plddt_per_atom.new_zeros(
            token_attention_mask.size(0),
            token_attention_mask.size(1),
        )
        atom_count = plddt_per_atom.new_zeros(
            token_attention_mask.size(0),
            token_attention_mask.size(1),
        )
        plddt_sum.scatter_add_(1, atom_to_token.clamp_min(0), plddt_per_atom * atom_values)
        atom_count.scatter_add_(1, atom_to_token.clamp_min(0), atom_values)
        categorical_plddt = plddt_sum / atom_count.clamp_min(1.0)
        scalar_plddt = 100.0 * torch.sigmoid(logits)
        predicted_plddt = 0.5 * (scalar_plddt + categorical_plddt)
        predicted_plddt = predicted_plddt * token_attention_mask.to(predicted_plddt.dtype)
        plddt_ca = plddt_per_atom.gather(
            dim=1,
            index=distogram_atom_idx.clamp_min(0),
        )
        plddt_ca = plddt_ca * token_attention_mask.to(plddt_ca.dtype)
        complex_plddt = (
            predicted_plddt * token_attention_mask.to(predicted_plddt.dtype)
        ).sum(dim=-1) / token_attention_mask.to(predicted_plddt.dtype).sum(dim=-1).clamp_min(1.0)

        resolved_logits = (
            atom_residue_features.unsqueeze(-2)
            @ resolved_weights
        ).squeeze(-2)
        resolved_logits = resolved_logits * atom_mask.to(resolved_logits.dtype)[..., None]
        resolved_probabilities = F.softmax(resolved_logits, dim=-1)
        resolved_probabilities = resolved_probabilities * atom_mask.to(
            resolved_probabilities.dtype
        )[..., None]
        pae_logits = self.pae_head(confidence_pair)
        pde_logits = self.pde_head(confidence_pair)
        pair_values = pair_mask.to(pae_logits.dtype)[..., None]
        pae_logits = pae_logits * pair_values
        pde_logits = pde_logits * pair_values
        pae_probabilities = F.softmax(pae_logits, dim=-1)
        pde_probabilities = F.softmax(pde_logits, dim=-1)
        pae_probabilities = pae_probabilities * pair_values
        pde_probabilities = pde_probabilities * pair_values
        predicted_aligned_error = (
            pae_probabilities * self.pae_error_centers
        ).sum(dim=-1)
        predicted_distance_error = (
            pde_probabilities * self.pde_error_centers
        ).sum(dim=-1)
        residue_count = token_attention_mask.to(pae_logits.dtype).sum(dim=-1, keepdim=True)
        d0 = 1.24 * (residue_count.clamp(min=19.0) - 15.0) ** (1.0 / 3.0) - 1.8
        tm_per_bin = 1.0 / (1.0 + (self.pae_error_centers[None, :] / d0) ** 2)
        tm_expected = (pae_probabilities * tm_per_bin[:, None, None, :]).sum(dim=-1)
        pair_values_scalar = pair_mask.to(tm_expected.dtype)
        tm_rows = (
            tm_expected * pair_values_scalar
        ).sum(dim=-1) / pair_values_scalar.sum(dim=-1).clamp_min(1.0)
        predicted_tm = tm_rows.max(dim=-1).values
        inter_chain_mask = asym_id[:, :, None] != asym_id[:, None, :]
        inter_chain_mask = inter_chain_mask & pair_mask
        inter_values = inter_chain_mask.to(tm_expected.dtype)
        iptm_rows = (
            tm_expected * inter_values
        ).sum(dim=-1) / inter_values.sum(dim=-1).clamp_min(1.0)
        predicted_iptm = torch.where(
            inter_chain_mask.any(dim=(1, 2)),
            iptm_rows.max(dim=-1).values,
            predicted_tm,
        )
        complex_iplddt = torch.where(
            inter_chain_mask.any(dim=(1, 2)),
            (
                predicted_plddt[:, :, None] * inter_values
            ).sum(dim=(1, 2)) / inter_values.sum(dim=(1, 2)).clamp_min(1.0),
            complex_plddt,
        )
        pair_chains_iptm = predicted_iptm[:, None, None]
        pair_summary = (
            confidence_pair * pair_values
        ).sum(dim=2) / pair_values.sum(dim=2).clamp_min(1.0)
        return {
            'logits': logits,
            'plddt_logits': plddt_logits,
            'plddt_probabilities': plddt_probabilities,
            'plddt_per_atom': plddt_per_atom,
            'predicted_plddt': predicted_plddt,
            'plddt_ca': plddt_ca,
            'complex_plddt': complex_plddt,
            'complex_iplddt': complex_iplddt,
            'pae_logits': pae_logits,
            'pae_probabilities': pae_probabilities,
            'predicted_aligned_error': predicted_aligned_error,
            'pde_logits': pde_logits,
            'pde_probabilities': pde_probabilities,
            'predicted_distance_error': predicted_distance_error,
            'resolved_logits': resolved_logits,
            'resolved_probabilities': resolved_probabilities,
            'predicted_tm': predicted_tm,
            'predicted_iptm': predicted_iptm,
            'pair_chains_iptm': pair_chains_iptm,
            'pair_summary': pair_summary,
            'confidence_pair': confidence_pair,
        }


def diffusion_coordinate_loss(predicted_atom_coords, target_atom_coords, atom_mask):
    squared_error = (predicted_atom_coords - target_atom_coords) ** 2
    return masked_mean(squared_error, atom_mask)


def pair_distance_loss(predicted_atom_coords, target_atom_coords, atom_mask, distogram_atom_idx, token_attention_mask):
    predicted_coords = gather_rep_atom_coords(predicted_atom_coords, distogram_atom_idx)
    target_coords = gather_rep_atom_coords(target_atom_coords, distogram_atom_idx)
    predicted_distances = pairwise_distances(predicted_coords)
    true_distances = pairwise_distances(target_coords)
    coord_mask = gather_rep_atom_mask(atom_mask, distogram_atom_idx) & token_attention_mask.bool()
    pair_mask = coordinate_pair_mask(coord_mask, remove_diagonal=True)
    loss_values = F.smooth_l1_loss(predicted_distances, true_distances, reduction='none')
    return (
        masked_mean(loss_values, pair_mask),
        predicted_distances,
        true_distances,
    )


def distogram_loss(logits, true_distances, coord_mask, boundaries):
    targets = torch.bucketize(true_distances.contiguous(), boundaries)
    loss_values = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        reduction='none',
    ).view_as(targets)
    pair_mask = coordinate_pair_mask(coord_mask, remove_diagonal=True)
    return masked_mean(loss_values, pair_mask)


def confidence_loss(confidence_logits, confidence_target, coord_mask):
    loss_values = (
        torch.sigmoid(confidence_logits) - confidence_target.detach()
    ) ** 2
    return masked_mean(loss_values, coord_mask)


def confidence_supervision_loss(
    confidence_output,
    predicted_atom_coords,
    target_atom_coords,
    atom_mask,
    reference_atom_mask,
    token_attention_mask,
    atom_to_token,
    distogram_atom_idx,
):
    true_atom_lddt = atom_lddt_per_atom_flat(
        predicted_atom_coords.detach(),
        target_atom_coords,
        atom_mask,
    ).detach()
    true_residue_lddt = scatter_atom_to_token(
        true_atom_lddt[..., None],
        atom_to_token,
        token_attention_mask.size(1),
        atom_mask,
    ).squeeze(-1)
    true_residue_lddt = true_residue_lddt * token_attention_mask.to(true_residue_lddt.dtype)
    scalar_loss = confidence_loss(
        confidence_output['logits'],
        true_residue_lddt,
        token_attention_mask,
    )

    plddt_bins = confidence_output['plddt_logits'].size(-1)
    plddt_targets = torch.clamp(
        (true_atom_lddt * plddt_bins).long(),
        0,
        plddt_bins - 1,
    )
    plddt_values = F.cross_entropy(
        confidence_output['plddt_logits'].reshape(-1, plddt_bins),
        plddt_targets.reshape(-1),
        reduction='none',
    ).view_as(plddt_targets)
    plddt_loss = masked_mean(plddt_values, atom_mask)

    pred_coords = gather_rep_atom_coords(predicted_atom_coords.detach(), distogram_atom_idx)
    true_coords = gather_rep_atom_coords(target_atom_coords, distogram_atom_idx)
    predicted_distances = pairwise_distances(pred_coords)
    true_distances = pairwise_distances(true_coords)
    distance_error = (predicted_distances - true_distances).abs().detach()
    error_boundaries = torch.linspace(
        0.0,
        32.0,
        confidence_output['pae_logits'].size(-1) - 1,
        device=distance_error.device,
        dtype=distance_error.dtype,
    )
    error_targets = torch.bucketize(distance_error.contiguous(), error_boundaries)
    rep_atom_mask = gather_rep_atom_mask(atom_mask, distogram_atom_idx)
    pair_mask = coordinate_pair_mask(rep_atom_mask & token_attention_mask.bool(), remove_diagonal=True)
    pae_bins = confidence_output['pae_logits'].size(-1)
    pae_values = F.cross_entropy(
        confidence_output['pae_logits'].reshape(-1, pae_bins),
        error_targets.reshape(-1),
        reduction='none',
    ).view_as(error_targets)
    pae_loss = masked_mean(pae_values, pair_mask)

    pde_bins = confidence_output['pde_logits'].size(-1)
    pde_values = F.cross_entropy(
        confidence_output['pde_logits'].reshape(-1, pde_bins),
        error_targets.reshape(-1).clamp(max=pde_bins - 1),
        reduction='none',
    ).view_as(error_targets)
    pde_loss = masked_mean(pde_values, pair_mask)

    resolved_targets = atom_mask.long()
    resolved_values = F.cross_entropy(
        confidence_output['resolved_logits'].reshape(-1, 2),
        resolved_targets.reshape(-1),
        reduction='none',
    ).view_as(resolved_targets)
    resolved_loss = masked_mean(resolved_values, reference_atom_mask)

    total = scalar_loss + 0.25 * (plddt_loss + pae_loss + pde_loss + resolved_loss)
    return total, {
        'residue_lddt': true_residue_lddt,
        'atom_lddt': true_atom_lddt,
        'scalar': scalar_loss,
        'plddt': plddt_loss,
        'pae': pae_loss,
        'pde': pde_loss,
        'resolved': resolved_loss,
    }


class ProteinFolding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.protein_lm = ProteinLM(self.config.protein_lm_config())
        self.inputs_embedder = InputsEmbedder(self.config)
        self.atom_input_encoder = self.inputs_embedder.atom_input_encoder
        self.language_model = LanguageModelShim(self.config)
        self.pair_embedder = PairEmbedder(self.config)
        self.recycling_trunk = RecyclingTrunk(self.config)
        self.distogram_head = DistogramHead(self.config)
        self.structure_denoiser = StructureDenoiser(self.config)
        self.confidence_head = ConfidenceHead(self.config)

    def encode_conditioning(
        self,
        input_ids,
        res_type,
        token_attention_mask,
        ref_pos,
        atom_attention_mask,
        ref_space_uid,
        atom_to_token,
        atom_element,
        atom_charge,
        atom_name_chars,
        atom_bonds,
        token_bonds,
        residue_index,
        token_index,
        asym_id,
        sym_id,
        entity_id,
        mol_type,
        recycling_generator=None,
        return_intermediates=False,
    ):
        lm_output = self.protein_lm(
            input_ids,
            token_attention_mask,
            asym_id=asym_id,
            residue_index=residue_index,
            mol_type=mol_type,
        )
        lm_pair = self.language_model(
            lm_output['hidden_states'],
            token_attention_mask,
            lm_dropout=self.training,
        )
        input_features = self.inputs_embedder(
            res_type,
            token_attention_mask,
            ref_pos,
            atom_attention_mask,
            ref_space_uid,
            atom_element,
            atom_charge,
            atom_name_chars,
            atom_to_token,
            atom_bonds,
            return_intermediates=return_intermediates,
        )
        (
            pair_initial,
            relative_position_features,
            token_bond_features,
            pair_embedder_intermediates,
        ) = self.pair_embedder(
            input_features['x_inputs'],
            token_attention_mask,
            token_bonds,
            residue_index,
            token_index,
            asym_id,
            sym_id,
            entity_id,
            return_intermediates=True,
        )
        pair_mask = sequence_pair_mask(token_attention_mask)
        pair, recycle_states, recycle_details = self.recycling_trunk(
            pair_initial,
            lm_pair,
            pair_mask,
            return_intermediates=return_intermediates,
            generator=recycling_generator,
        )
        distogram_logits = self.distogram_head(pair, pair_mask)

        output = {
            'pair_initial': pair_initial,
            'pair': pair,
            'pair_mask': pair_mask,
            'atom_features': input_features['atom_features'],
            'x_inputs': input_features['x_inputs'],
            'relative_position_encoding': relative_position_features,
            'token_bond_features': token_bond_features,
            'distogram_logits': distogram_logits,
        }
        if return_intermediates:
            output['intermediates'] = {
                'lm_hidden_states': lm_output['hidden_states'],
                'lm_input_ids': lm_output['lm_input_ids'],
                'lm_attention_mask': lm_output['lm_attention_mask'],
                'lm_sequence_id': lm_output['sequence_id'],
                'residue_to_lm_index': lm_output['residue_to_lm_index'],
                'lm_pair': lm_pair,
                'inputs_embedder': input_features,
                'pair_embedder': pair_embedder_intermediates,
                'pair_recycles': recycle_states,
                'pair_blocks': recycle_details,
            }
        return output

    def forward(
        self,
        *,
        token_index,
        residue_index,
        asym_id,
        sym_id,
        entity_id,
        mol_type,
        input_ids,
        res_type,
        token_attention_mask,
        token_bonds,
        ref_pos,
        atom_attention_mask,
        ref_space_uid,
        atom_to_token,
        atom_bonds,
        atom_mask,
        atom_element,
        atom_charge,
        atom_name_chars,
        distogram_atom_idx,
        target_atom_coords=None,
        noisy_atom_coords=None,
        sigma=None,
        noise=None,
        return_intermediates=False,
    ):
        batch_size, residue_length = input_ids.shape
        assert input_ids.shape == res_type.shape == token_attention_mask.shape
        assert residue_length <= self.config.block_size
        assert ref_pos.ndim == 3 and ref_pos.size(-1) == 3
        assert atom_mask.shape == atom_attention_mask.shape == atom_to_token.shape

        token_attention_mask = token_attention_mask.bool()
        atom_attention_mask = atom_attention_mask.bool() & (atom_to_token >= 0)
        coordinate_atom_mask = atom_mask.bool() & atom_attention_mask
        safe_atom_to_token = torch.where(
            atom_attention_mask,
            atom_to_token.clamp_min(0),
            torch.zeros_like(atom_to_token),
        )
        conditioning = self.encode_conditioning(
            input_ids,
            res_type,
            token_attention_mask,
            ref_pos,
            atom_attention_mask,
            ref_space_uid,
            safe_atom_to_token,
            atom_element,
            atom_charge,
            atom_name_chars,
            atom_bonds,
            token_bonds,
            residue_index,
            token_index,
            asym_id,
            sym_id,
            entity_id,
            mol_type,
            return_intermediates=return_intermediates,
        )

        centered_target = None
        if target_atom_coords is not None:
            assert target_atom_coords.shape == coordinate_atom_mask.shape + (3,)
            centered_target = masked_center(target_atom_coords, coordinate_atom_mask)

        if sigma is None:
            assert centered_target is not None
            sigma = sample_log_uniform_sigmas(
                conditioning['x_inputs'],
                self.config.sigma_min,
                self.config.sigma_max,
            )
        assert sigma.shape == (batch_size,)
        sigma = sigma.to(conditioning['x_inputs'])

        if noisy_atom_coords is None:
            assert centered_target is not None
            if noise is None:
                noise = centered_gaussian_noise(centered_target, atom_attention_mask)
            else:
                noise = masked_center(
                    noise.to(centered_target.dtype),
                    atom_attention_mask,
                )
            noisy_atom_coords = centered_target + sigma[:, None, None] * noise
        else:
            assert noisy_atom_coords.shape == atom_attention_mask.shape + (3,)
            noisy_atom_coords = masked_center(
                noisy_atom_coords.to(conditioning['x_inputs'].dtype),
                atom_attention_mask,
            )
            if noise is not None:
                noise = masked_center(
                    noise.to(conditioning['x_inputs'].dtype),
                    atom_attention_mask,
                )

        denoiser_output = self.structure_denoiser(
            noisy_atom_coords,
            conditioning['x_inputs'],
            conditioning['pair'],
            conditioning['relative_position_encoding'],
            ref_pos,
            atom_attention_mask,
            ref_space_uid,
            atom_element,
            atom_charge,
            atom_name_chars,
            safe_atom_to_token,
            atom_bonds,
            sigma,
            token_attention_mask,
            atom_attention_mask,
            return_intermediates=return_intermediates,
        )
        pred_atom_coords = denoiser_output['pred_atom_coords']

        detached_pred_atom_coords = pred_atom_coords.detach()
        confidence_output = self.confidence_head(
            conditioning['x_inputs'],
            conditioning['pair'],
            detached_pred_atom_coords,
            distogram_atom_idx,
            token_attention_mask,
            safe_atom_to_token,
            atom_attention_mask,
            asym_id,
            mol_type,
            relative_position_encoding=conditioning['relative_position_encoding'],
            token_bond_features=conditioning['token_bond_features'],
        )

        losses = {
            'diffusion': None,
            'distance': None,
            'distogram': None,
            'confidence': None,
        }
        total_loss = None
        true_distances = None
        predicted_distances = pairwise_distances(
            gather_rep_atom_coords(pred_atom_coords, distogram_atom_idx)
        )
        true_plddt = None
        confidence_targets = None

        if centered_target is not None:
            losses['diffusion'] = diffusion_coordinate_loss(
                pred_atom_coords,
                centered_target,
                coordinate_atom_mask,
            )
            (
                losses['distance'],
                predicted_distances,
                true_distances,
            ) = pair_distance_loss(
                pred_atom_coords,
                centered_target,
                coordinate_atom_mask,
                distogram_atom_idx,
                token_attention_mask,
            )
            distogram_coord_mask = (
                gather_rep_atom_mask(coordinate_atom_mask, distogram_atom_idx)
                & token_attention_mask
            )
            losses['distogram'] = distogram_loss(
                conditioning['distogram_logits'],
                true_distances,
                distogram_coord_mask,
                self.distogram_head.boundaries,
            )
            true_plddt = scatter_atom_to_token(
                atom_lddt_per_atom_flat(
                    detached_pred_atom_coords,
                    centered_target,
                    coordinate_atom_mask,
                ).detach()[..., None],
                safe_atom_to_token,
                token_attention_mask.size(1),
                coordinate_atom_mask,
            ).squeeze(-1)
            true_plddt = true_plddt * token_attention_mask.to(true_plddt.dtype)
            losses['confidence'], confidence_targets = confidence_supervision_loss(
                confidence_output,
                detached_pred_atom_coords,
                centered_target,
                coordinate_atom_mask,
                atom_attention_mask,
                token_attention_mask,
                safe_atom_to_token,
                distogram_atom_idx,
            )
            total_loss = (
                losses['diffusion']
                + 0.5 * losses['distance']
                + 0.2 * losses['distogram']
                + 0.1 * losses['confidence']
            )

        output = {
            'pred_atom_coords': pred_atom_coords,
            'distogram_logits': conditioning['distogram_logits'],
            'predicted_plddt': confidence_output['predicted_plddt'],
            'confidence_logits': confidence_output['logits'],
            'plddt_logits': confidence_output['plddt_logits'],
            'plddt_probabilities': confidence_output['plddt_probabilities'],
            'plddt_per_atom': confidence_output['plddt_per_atom'],
            'plddt_ca': confidence_output['plddt_ca'],
            'complex_plddt': confidence_output['complex_plddt'],
            'complex_iplddt': confidence_output['complex_iplddt'],
            'pae_logits': confidence_output['pae_logits'],
            'pae_probabilities': confidence_output['pae_probabilities'],
            'pde_logits': confidence_output['pde_logits'],
            'pde_probabilities': confidence_output['pde_probabilities'],
            'predicted_aligned_error': confidence_output['predicted_aligned_error'],
            'predicted_distance_error': confidence_output['predicted_distance_error'],
            'resolved_logits': confidence_output['resolved_logits'],
            'resolved_probabilities': confidence_output['resolved_probabilities'],
            'predicted_tm': confidence_output['predicted_tm'],
            'predicted_iptm': confidence_output['predicted_iptm'],
            'pair_chains_iptm': confidence_output['pair_chains_iptm'],
            'loss': total_loss,
            'losses': losses,
        }
        if return_intermediates:
            output['intermediates'] = {
                **conditioning['intermediates'],
                'pair_initial': conditioning['pair_initial'],
                'pair': conditioning['pair'],
                'relative_position_encoding': conditioning['relative_position_encoding'],
                'token_bond_features': conditioning['token_bond_features'],
                'sigma': sigma,
                'noise': noise,
                'noisy_atom_coords': noisy_atom_coords,
                'denoiser': denoiser_output.get('intermediates'),
                'denoiser_nodes': denoiser_output['nodes'],
                'true_distances': true_distances,
                'predicted_distances': predicted_distances,
                'true_plddt': true_plddt,
                'confidence_targets': confidence_targets,
                'confidence_pair_summary': confidence_output['pair_summary'],
                'confidence_pair': confidence_output['confidence_pair'],
                'resolved_logits': confidence_output['resolved_logits'],
                'resolved_probabilities': confidence_output['resolved_probabilities'],
            }
        return output

    @torch.no_grad()
    def sample(
        self,
        *,
        token_index,
        residue_index,
        asym_id,
        sym_id,
        entity_id,
        mol_type,
        input_ids,
        res_type,
        token_attention_mask,
        token_bonds,
        ref_pos,
        atom_attention_mask,
        ref_space_uid,
        atom_to_token,
        atom_bonds,
        atom_mask,
        atom_element,
        atom_charge,
        atom_name_chars,
        distogram_atom_idx,
        num_steps=None,
        seed=0,
        return_trajectory=False,
        return_confidence_inputs=False,
    ):
        batch_size, residue_length = input_ids.shape
        token_attention_mask = token_attention_mask.bool()
        atom_attention_mask = atom_attention_mask.bool() & (atom_to_token >= 0)
        atom_mask = atom_mask.bool() & atom_attention_mask
        generation_atom_mask = atom_attention_mask
        safe_atom_to_token = torch.where(
            atom_attention_mask,
            atom_to_token.clamp_min(0),
            torch.zeros_like(atom_to_token),
        )
        num_steps = num_steps or self.config.sampling_steps
        generator = torch.Generator(device=input_ids.device)
        generator.manual_seed(seed)

        conditioning = self.encode_conditioning(
            input_ids,
            res_type,
            token_attention_mask,
            ref_pos,
            atom_attention_mask,
            ref_space_uid,
            safe_atom_to_token,
            atom_element,
            atom_charge,
            atom_name_chars,
            atom_bonds,
            token_bonds,
            residue_index,
            token_index,
            asym_id,
            sym_id,
            entity_id,
            mol_type,
            recycling_generator=generator,
            return_intermediates=False,
        )
        coordinate_reference = conditioning['x_inputs'].new_zeros(
            input_ids.size(0), generation_atom_mask.size(1), 3
        )

        noise = centered_gaussian_noise(
            coordinate_reference, generation_atom_mask, generator=generator
        )
        atom_coords = self.config.sigma_max * noise
        noise_schedule = make_sampling_schedule(
            conditioning['x_inputs'],
            self.config.sigma_min,
            self.config.sigma_max,
            num_steps,
        )
        trajectory = [atom_coords.clone()] if return_trajectory else None
        final_nodes = None
        previous_denoised = None

        for step_index in range(num_steps):
            sigma_value = noise_schedule[step_index]
            next_sigma = noise_schedule[step_index + 1]
            atom_coords, previous_denoised = center_random_augmentation(
                atom_coords,
                generation_atom_mask,
                generator=generator,
                second_coords=previous_denoised,
            )
            gamma = 0.2 if sigma_value > self.config.sigma_min else 0.0
            t_hat = sigma_value * (1.0 + gamma)
            churn_scale = torch.sqrt((t_hat * t_hat - sigma_value * sigma_value).clamp_min(0.0))
            if float(churn_scale.item()) > 0.0:
                atom_coords = atom_coords + churn_scale * centered_gaussian_noise(
                    atom_coords,
                    generation_atom_mask,
                    generator=generator,
                )
            sigma = t_hat.expand(input_ids.size(0))
            denoiser_output = self.structure_denoiser(
                atom_coords,
                conditioning['x_inputs'],
                conditioning['pair'],
                conditioning['relative_position_encoding'],
                ref_pos,
                atom_attention_mask,
                ref_space_uid,
                atom_element,
                atom_charge,
                atom_name_chars,
                safe_atom_to_token,
                atom_bonds,
                sigma,
                token_attention_mask,
                generation_atom_mask,
                return_intermediates=False,
            )
            pred_atom_coords = denoiser_output['pred_atom_coords']
            aligned_noisy = weighted_rigid_align(atom_coords, pred_atom_coords, generation_atom_mask)
            estimated_noise = (
                aligned_noisy - pred_atom_coords
            ) / t_hat.clamp_min(1e-8)
            atom_coords = aligned_noisy + (next_sigma - t_hat) * estimated_noise
            atom_coords = masked_center(atom_coords, generation_atom_mask)
            previous_denoised = pred_atom_coords
            final_nodes = denoiser_output['nodes']
            if return_trajectory:
                trajectory.append(atom_coords.clone())

        confidence_output = self.confidence_head(
            conditioning['x_inputs'],
            conditioning['pair'],
            atom_coords,
            distogram_atom_idx,
            token_attention_mask,
            safe_atom_to_token,
            generation_atom_mask,
            asym_id,
            mol_type,
            relative_position_encoding=conditioning['relative_position_encoding'],
            token_bond_features=conditioning['token_bond_features'],
        )
        output = {
            'atom_coords': atom_coords,
            'predicted_plddt': confidence_output['predicted_plddt'],
            'confidence_logits': confidence_output['logits'],
            'plddt_logits': confidence_output['plddt_logits'],
            'plddt_probabilities': confidence_output['plddt_probabilities'],
            'plddt_per_atom': confidence_output['plddt_per_atom'],
            'plddt_ca': confidence_output['plddt_ca'],
            'complex_plddt': confidence_output['complex_plddt'],
            'complex_iplddt': confidence_output['complex_iplddt'],
            'pae_logits': confidence_output['pae_logits'],
            'pae_probabilities': confidence_output['pae_probabilities'],
            'predicted_aligned_error': confidence_output['predicted_aligned_error'],
            'pde_logits': confidence_output['pde_logits'],
            'pde_probabilities': confidence_output['pde_probabilities'],
            'predicted_distance_error': confidence_output['predicted_distance_error'],
            'resolved_logits': confidence_output['resolved_logits'],
            'resolved_probabilities': confidence_output['resolved_probabilities'],
            'predicted_tm': confidence_output['predicted_tm'],
            'predicted_iptm': confidence_output['predicted_iptm'],
            'pair_chains_iptm': confidence_output['pair_chains_iptm'],
            'distogram_logits': conditioning['distogram_logits'],
            'distogram_probabilities': (
                F.softmax(conditioning['distogram_logits'], dim=-1)
                * conditioning['pair_mask'].to(conditioning['distogram_logits'].dtype)[..., None]
            ),
            'schedule': noise_schedule,
        }
        if return_trajectory:
            output['trajectory'] = trajectory
        if return_confidence_inputs:
            output['confidence_inputs'] = {
                'x_inputs': conditioning['x_inputs'].detach(),
                'nodes': final_nodes.detach(),
                'pair': conditioning['pair'].detach(),
                'pair_initial': conditioning['pair_initial'].detach(),
                'relative_position_encoding': (
                    conditioning['relative_position_encoding'].detach()
                    if conditioning['relative_position_encoding'] is not None
                    else None
                ),
                'token_bond_features': (
                    conditioning['token_bond_features'].detach()
                    if conditioning['token_bond_features'] is not None
                    else None
                ),
                'atom_coords': atom_coords.detach(),
                'residue_mask': token_attention_mask,
                'atom_mask': generation_atom_mask,
                'target_atom_mask': atom_mask,
                'reference_atom_mask': atom_attention_mask,
                'atom_to_token': safe_atom_to_token,
                'atom_bonds': atom_bonds,
                'distogram_atom_idx': distogram_atom_idx,
                'asym_id': asym_id,
                'mol_type': mol_type,
            }
        return output


class ProteinDataLoaderLite:
    def __init__(
        self,
        data_root,
        batch_size,
        block_size=64,
        split='train',
        training=None,
        random_rotation=True,
        sequence_only=False,
        seed=1337,
    ):
        assert split in {'train', 'val', 'test'}
        self.batch_size = batch_size
        self.block_size = block_size
        self.split = split
        self.training = split == 'train' if training is None else training
        self.random_rotation = random_rotation and self.training
        self.sequence_only = sequence_only
        self.random = random.Random(seed)
        self.torch_generator = torch.Generator().manual_seed(seed)
        self.shards = sorted((Path(data_root) / split).glob('*.npz'))
        if not self.shards:
            raise FileNotFoundError(f'no .npz shards found in {Path(data_root) / split}')
        self.reset()

    def reset(self):
        self.shard_order = list(self.shards)
        if self.training:
            self.random.shuffle(self.shard_order)
        self.shard_index = 0
        self.example_index = 0
        self._load_current_shard()

    def _load_current_shard(self):
        import numpy as np

        with np.load(
            self.shard_order[self.shard_index],
            allow_pickle=False,
        ) as shard:
            self.current = {
                'input_ids': torch.tensor(shard['input_ids'], dtype=torch.long),
                'residue_mask': torch.tensor(shard['residue_mask'], dtype=torch.bool),
                'lengths': torch.tensor(shard['lengths'], dtype=torch.long),
            }
            if not self.sequence_only:
                self.current.update(
                    {
                        'res_type': torch.tensor(shard['res_type'], dtype=torch.long),
                        'atom_coords': torch.tensor(shard['atom_coords'], dtype=torch.float32),
                        'atom_mask': torch.tensor(shard['atom_mask'], dtype=torch.bool),
                        'reference_atom_coords': torch.tensor(
                            shard['reference_atom_coords'],
                            dtype=torch.float32,
                        ),
                        'reference_atom_mask': torch.tensor(
                            shard['reference_atom_mask'],
                            dtype=torch.bool,
                        ),
                        'atom_to_token': torch.tensor(shard['atom_to_token'], dtype=torch.long),
                        'atom_element': torch.tensor(shard['atom_element'], dtype=torch.long),
                        'atom_charge': torch.tensor(shard['atom_charge'], dtype=torch.float32),
                        'atom_name_chars': torch.tensor(
                            shard['atom_name_chars'],
                            dtype=torch.long,
                        ),
                        'residue_atom_bonds': torch.tensor(
                            shard['residue_atom_bonds'],
                            dtype=torch.bool,
                        ),
                        'peptide_bond_mask': torch.tensor(
                            shard['peptide_bond_mask'],
                            dtype=torch.bool,
                        ),
                    }
                )
                for key in (
                    'token_bonds',
                    'ref_space_uid',
                    'residue_index',
                    'token_index',
                    'asym_id',
                    'sym_id',
                    'entity_id',
                    'mol_type',
                ):
                    if key in shard.files:
                        dtype = torch.bool if key == 'token_bonds' else torch.long
                        self.current[key] = torch.tensor(shard[key], dtype=dtype)
        self.example_order = list(range(self.current['input_ids'].size(0)))
        if self.training:
            self.random.shuffle(self.example_order)
        self.example_index = 0

    def _next_example(self):
        if self.example_index >= len(self.example_order):
            self.shard_index += 1
            if self.shard_index >= len(self.shard_order):
                self.reset()
            else:
                self._load_current_shard()

        index = self.example_order[self.example_index]
        self.example_index += 1
        length = int(self.current['lengths'][index].item())
        if length > self.block_size:
            if self.training:
                crop_start = self.random.randint(0, length - self.block_size)
            else:
                crop_start = (length - self.block_size) // 2
            crop_end = crop_start + self.block_size
        else:
            crop_start = 0
            crop_end = length

        example = {}
        for key, value in self.current.items():
            if key == 'lengths':
                continue
            if key == 'token_bonds':
                example[key] = value[index, crop_start:crop_end, crop_start:crop_end].clone()
            else:
                example[key] = value[index, crop_start:crop_end].clone()
        return localize_atom_token_indices(example, crop_start)

    def next_batch(self, device=None):
        examples = [self._next_example() for _ in range(self.batch_size)]
        if self.sequence_only:
            batch = {
                'input_ids': torch.full(
                    (self.batch_size, self.block_size),
                    ESM_PAD_ID,
                    dtype=torch.long,
                ),
                'residue_mask': torch.zeros(
                    self.batch_size,
                    self.block_size,
                    dtype=torch.bool,
                ),
            }
            for batch_index, example in enumerate(examples):
                length = min(example['input_ids'].size(0), self.block_size)
                batch['input_ids'][batch_index, :length] = example['input_ids'][
                    :length
                ]
                batch['residue_mask'][batch_index, :length] = example[
                    'residue_mask'
                ][:length]
            if device is not None:
                batch = {key: value.to(device) for key, value in batch.items()}
            return batch

        batch = {
            'input_ids': torch.full(
                (self.batch_size, self.block_size), ESM_PAD_ID, dtype=torch.long
            ),
            'res_type': torch.full(
                (self.batch_size, self.block_size),
                RES_PAD_ID,
                dtype=torch.long,
            ),
            'residue_mask': torch.zeros(self.batch_size, self.block_size, dtype=torch.bool),
            'atom_coords': torch.zeros(
                self.batch_size,
                self.block_size,
                len(ATOM_SLOTS),
                3,
                dtype=torch.float32,
            ),
            'atom_mask': torch.zeros(
                self.batch_size,
                self.block_size,
                len(ATOM_SLOTS),
                dtype=torch.bool,
            ),
            'reference_atom_coords': torch.zeros(
                self.batch_size,
                self.block_size,
                len(ATOM_SLOTS),
                3,
                dtype=torch.float32,
            ),
            'reference_atom_mask': torch.zeros(
                self.batch_size,
                self.block_size,
                len(ATOM_SLOTS),
                dtype=torch.bool,
            ),
            'atom_to_token': torch.full(
                (self.batch_size, self.block_size, len(ATOM_SLOTS)),
                -1,
                dtype=torch.long,
            ),
            'atom_element': torch.zeros(
                self.batch_size,
                self.block_size,
                len(ATOM_SLOTS),
                dtype=torch.long,
            ),
            'atom_charge': torch.zeros(
                self.batch_size,
                self.block_size,
                len(ATOM_SLOTS),
                dtype=torch.float32,
            ),
            'atom_name_chars': torch.zeros(
                self.batch_size,
                self.block_size,
                len(ATOM_SLOTS),
                ATOM_NAME_CHAR_WIDTH,
                dtype=torch.long,
            ),
            'residue_atom_bonds': torch.zeros(
                self.batch_size,
                self.block_size,
                len(ATOM_SLOTS),
                len(ATOM_SLOTS),
                dtype=torch.bool,
            ),
            'peptide_bond_mask': torch.zeros(
                self.batch_size,
                self.block_size,
                dtype=torch.bool,
            ),
            'token_bonds': torch.zeros(
                self.batch_size,
                self.block_size,
                self.block_size,
                dtype=torch.bool,
            ),
            'ref_space_uid': torch.zeros(
                self.batch_size,
                self.block_size,
                len(ATOM_SLOTS),
                dtype=torch.long,
            ),
            'residue_index': torch.zeros(self.batch_size, self.block_size, dtype=torch.long),
            'token_index': torch.zeros(self.batch_size, self.block_size, dtype=torch.long),
            'asym_id': torch.zeros(self.batch_size, self.block_size, dtype=torch.long),
            'sym_id': torch.zeros(self.batch_size, self.block_size, dtype=torch.long),
            'entity_id': torch.zeros(self.batch_size, self.block_size, dtype=torch.long),
            'mol_type': torch.zeros(self.batch_size, self.block_size, dtype=torch.long),
        }

        for batch_index, example in enumerate(examples):
            length = min(example['input_ids'].size(0), self.block_size)
            for key in batch:
                if key not in example:
                    if key in {'residue_index', 'token_index'}:
                        batch[key][batch_index, :length] = torch.arange(length)
                    continue
                if key == 'token_bonds':
                    batch[key][batch_index, :length, :length] = example[key][:length, :length]
                else:
                    batch[key][batch_index, :length] = example[key][:length]

        batch['atom_mask'] = batch['atom_mask'] & batch['residue_mask'][:, :, None]
        batch['reference_atom_mask'] = (
            batch['reference_atom_mask'] & batch['residue_mask'][:, :, None]
        )
        batch['residue_atom_bonds'] = (
            batch['residue_atom_bonds']
            & batch['reference_atom_mask'][:, :, :, None]
            & batch['reference_atom_mask'][:, :, None, :]
        )
        batch['peptide_bond_mask'] = batch['peptide_bond_mask'] & batch['residue_mask']
        batch['peptide_bond_mask'][:, -1] = False
        batch['token_bonds'] = torch.zeros_like(batch['token_bonds'])
        batch['atom_coords'] = masked_center(batch['atom_coords'], batch['atom_mask'])
        if self.random_rotation:
            rotation = random_rotation_matrix(
                batch['atom_coords'],
                generator=self.torch_generator,
            )
            batch['atom_coords'] = batch['atom_coords'] @ rotation[:, None, :, :]
            batch['atom_coords'] = masked_center(batch['atom_coords'], batch['atom_mask'])
        batch = add_atom_token_fields(batch)
        if device is not None:
            batch = {key: value.to(device) for key, value in batch.items()}
        return batch


def _atom_element_id(atom_name):
    if atom_name.startswith('C'):
        return 6
    if atom_name.startswith('N'):
        return 7
    if atom_name.startswith('O'):
        return 8
    if atom_name.startswith('S'):
        return 16
    return 0


def _atom_name_chars(atom_name):
    values = torch.zeros(ATOM_NAME_CHAR_WIDTH, dtype=torch.long)
    padded = atom_name.ljust(ATOM_NAME_CHAR_WIDTH)[:ATOM_NAME_CHAR_WIDTH]
    for index, value in enumerate(padded):
        values[index] = 0 if value == ' ' else ord(value) - 32
    return values


def make_atom_feature_batch(sequences, block_size=64, device=None):
    tokenizer = ProteinTokenizer()
    atom_count = len(ATOM_SLOTS)
    input_ids = torch.full((len(sequences), block_size), ESM_PAD_ID, dtype=torch.long)
    res_type = torch.full((len(sequences), block_size), RES_PAD_ID, dtype=torch.long)
    residue_mask = torch.zeros(len(sequences), block_size, dtype=torch.bool)
    atom_coords = torch.zeros(len(sequences), block_size, atom_count, 3)
    atom_mask = torch.zeros(len(sequences), block_size, atom_count, dtype=torch.bool)
    reference_atom_coords = torch.zeros(len(sequences), block_size, atom_count, 3)
    reference_atom_mask = torch.zeros(len(sequences), block_size, atom_count, dtype=torch.bool)
    atom_to_token = torch.full((len(sequences), block_size, atom_count), -1, dtype=torch.long)
    atom_element = torch.zeros(len(sequences), block_size, atom_count, dtype=torch.long)
    atom_charge = torch.zeros(len(sequences), block_size, atom_count)
    atom_name_chars = torch.zeros(
        len(sequences),
        block_size,
        atom_count,
        ATOM_NAME_CHAR_WIDTH,
        dtype=torch.long,
    )
    residue_atom_bonds = torch.zeros(
        len(sequences),
        block_size,
        atom_count,
        atom_count,
        dtype=torch.bool,
    )
    peptide_bond_mask = torch.zeros(len(sequences), block_size, dtype=torch.bool)
    synthetic_atoms = ('N', 'CA', 'C', 'O', 'CB')
    synthetic_offsets = {
        'N': torch.tensor([-1.25, 0.0, 0.0]),
        'CA': torch.tensor([0.0, 0.0, 0.0]),
        'C': torch.tensor([1.52, 0.0, 0.0]),
        'O': torch.tensor([2.10, 0.85, 0.0]),
        'CB': torch.tensor([0.0, 1.52, 0.0]),
    }

    for batch_index, sequence in enumerate(sequences):
        ids, types = tokenizer.encode(sequence)
        assert 0 < len(ids) <= block_size
        input_ids[batch_index, :len(ids)] = torch.tensor(ids)
        res_type[batch_index, :len(ids)] = torch.tensor(types)
        residue_mask[batch_index, :len(ids)] = True
        if len(ids) > 1:
            peptide_bond_mask[batch_index, : len(ids) - 1] = True
        for residue_index, residue in enumerate(sequence):
            atoms_for_residue = synthetic_atoms if residue != 'G' else synthetic_atoms[:4]
            for atom_name in atoms_for_residue:
                atom_index = ATOM_SLOT_TO_INDEX[atom_name]
                atom_mask[batch_index, residue_index, atom_index] = True
                reference_atom_mask[batch_index, residue_index, atom_index] = True
                reference_atom_coords[batch_index, residue_index, atom_index] = (
                    synthetic_offsets[atom_name]
                )
                atom_to_token[batch_index, residue_index, atom_index] = residue_index
                atom_element[batch_index, residue_index, atom_index] = _atom_element_id(atom_name)
                atom_name_chars[batch_index, residue_index, atom_index] = (
                    _atom_name_chars(atom_name)
                )
            for atom_a, atom_b in (('N', 'CA'), ('CA', 'C'), ('C', 'O'), ('CA', 'CB')):
                if atom_a not in atoms_for_residue or atom_b not in atoms_for_residue:
                    continue
                index_a = ATOM_SLOT_TO_INDEX[atom_a]
                index_b = ATOM_SLOT_TO_INDEX[atom_b]
                residue_atom_bonds[batch_index, residue_index, index_a, index_b] = True
                residue_atom_bonds[batch_index, residue_index, index_b, index_a] = True

    batch = {
        'input_ids': input_ids,
        'res_type': res_type,
        'residue_mask': residue_mask,
        'atom_coords': atom_coords,
        'atom_mask': atom_mask,
        'reference_atom_coords': reference_atom_coords,
        'reference_atom_mask': reference_atom_mask,
        'atom_to_token': atom_to_token,
        'atom_element': atom_element,
        'atom_charge': atom_charge,
        'atom_name_chars': atom_name_chars,
        'residue_atom_bonds': residue_atom_bonds,
        'peptide_bond_mask': peptide_bond_mask,
    }
    batch = add_atom_token_fields(batch)
    if device is not None:
        batch = {key: value.to(device) for key, value in batch.items()}
    return batch


def make_synthetic_batch(batch_size=4, block_size=32, min_length=12, device=None, seed=1337):
    assert 2 <= min_length <= block_size

    tokenizer = ProteinTokenizer()
    generator = torch.Generator().manual_seed(seed)
    input_ids = torch.full(
        (batch_size, block_size),
        ESM_PAD_ID,
        dtype=torch.long,
    )
    res_type = torch.full(
        (batch_size, block_size),
        RES_PAD_ID,
        dtype=torch.long,
    )
    residue_mask = torch.zeros(batch_size, block_size, dtype=torch.bool)
    atom_count = len(ATOM_SLOTS)
    atom_coords = torch.zeros(batch_size, block_size, atom_count, 3)
    atom_mask = torch.zeros(batch_size, block_size, atom_count, dtype=torch.bool)
    reference_atom_coords = torch.zeros(batch_size, block_size, atom_count, 3)
    reference_atom_mask = torch.zeros(batch_size, block_size, atom_count, dtype=torch.bool)
    atom_to_token = torch.full((batch_size, block_size, atom_count), -1, dtype=torch.long)
    atom_element = torch.zeros(batch_size, block_size, atom_count, dtype=torch.long)
    atom_charge = torch.zeros(batch_size, block_size, atom_count)
    atom_name_chars = torch.zeros(
        batch_size,
        block_size,
        atom_count,
        ATOM_NAME_CHAR_WIDTH,
        dtype=torch.long,
    )
    residue_atom_bonds = torch.zeros(
        batch_size,
        block_size,
        atom_count,
        atom_count,
        dtype=torch.bool,
    )
    peptide_bond_mask = torch.zeros(batch_size, block_size, dtype=torch.bool)
    synthetic_atoms = ('N', 'CA', 'C', 'O', 'CB')
    synthetic_offsets = {
        'N': torch.tensor([-1.25, 0.0, 0.0]),
        'CA': torch.tensor([0.0, 0.0, 0.0]),
        'C': torch.tensor([1.52, 0.0, 0.0]),
        'O': torch.tensor([2.10, 0.85, 0.0]),
        'CB': torch.tensor([0.0, 1.52, 0.0]),
    }
    synthetic_bonds = (('N', 'CA'), ('CA', 'C'), ('C', 'O'), ('CA', 'CB'))
    patterns = ('ACDEFGHIKLMNPQRSTVWY', 'GAVLIPFYWSTCMNQDEKRH')

    for batch_index in range(batch_size):
        span = block_size - min_length + 1
        length = min_length + ((batch_index * 7) % span)
        sequence = ''.join(
            patterns[batch_index % len(patterns)][position % 20]
            for position in range(length)
        )
        ids, types = tokenizer.encode(sequence)
        input_ids[batch_index, :length] = torch.tensor(ids)
        res_type[batch_index, :length] = torch.tensor(types)
        residue_mask[batch_index, :length] = True
        if length > 1:
            peptide_bond_mask[batch_index, : length - 1] = True

        position = torch.arange(length, dtype=torch.float32)
        shape_kind = batch_index % 3
        if shape_kind == 0:
            angle = position * math.radians(100.0)
            trace = torch.stack(
                [
                    2.3 * torch.cos(angle),
                    2.3 * torch.sin(angle),
                    1.5 * position,
                ],
                dim=-1,
            )
        elif shape_kind == 1:
            trace = torch.stack(
                [
                    3.8 * position,
                    1.2 * torch.where(
                        (position.long() % 2) == 0,
                        torch.ones_like(position),
                        -torch.ones_like(position),
                    ),
                    torch.zeros_like(position),
                ],
                dim=-1,
            )
        else:
            steps = torch.randn(length, 3, generator=generator)
            steps = 3.8 * steps / torch.sqrt(
                (steps * steps).sum(dim=-1, keepdim=True) + 1e-8
            )
            trace = torch.cumsum(steps, dim=0)
        for residue_index, residue in enumerate(sequence):
            atoms_for_residue = synthetic_atoms if residue != 'G' else synthetic_atoms[:4]
            for atom_name in atoms_for_residue:
                atom_index = ATOM_SLOT_TO_INDEX[atom_name]
                offset = synthetic_offsets[atom_name]
                atom_coords[batch_index, residue_index, atom_index] = (
                    trace[residue_index] + offset
                )
                atom_mask[batch_index, residue_index, atom_index] = True
                reference_atom_coords[batch_index, residue_index, atom_index] = offset
                reference_atom_mask[batch_index, residue_index, atom_index] = True
                atom_to_token[batch_index, residue_index, atom_index] = residue_index
                atom_element[batch_index, residue_index, atom_index] = _atom_element_id(
                    atom_name
                )
                atom_name_chars[batch_index, residue_index, atom_index] = (
                    _atom_name_chars(atom_name)
                )
            for atom_a, atom_b in synthetic_bonds:
                if atom_a not in atoms_for_residue or atom_b not in atoms_for_residue:
                    continue
                index_a = ATOM_SLOT_TO_INDEX[atom_a]
                index_b = ATOM_SLOT_TO_INDEX[atom_b]
                residue_atom_bonds[batch_index, residue_index, index_a, index_b] = True
                residue_atom_bonds[batch_index, residue_index, index_b, index_a] = True

    atom_coords = masked_center(atom_coords, atom_mask)
    batch = {
        'input_ids': input_ids,
        'res_type': res_type,
        'residue_mask': residue_mask,
        'atom_coords': atom_coords,
        'atom_mask': atom_mask,
        'reference_atom_coords': reference_atom_coords,
        'reference_atom_mask': reference_atom_mask,
        'atom_to_token': atom_to_token,
        'atom_element': atom_element,
        'atom_charge': atom_charge,
        'atom_name_chars': atom_name_chars,
        'residue_atom_bonds': residue_atom_bonds,
        'peptide_bond_mask': peptide_bond_mask,
    }
    batch = add_atom_token_fields(batch)
    if device is not None:
        batch = {key: value.to(device) for key, value in batch.items()}
    return batch


def folding_model_inputs(batch):
    return {
        'token_index': batch['token_index'],
        'residue_index': batch['residue_index'],
        'asym_id': batch['asym_id'],
        'sym_id': batch['sym_id'],
        'entity_id': batch['entity_id'],
        'mol_type': batch['mol_type'],
        'input_ids': batch['input_ids'],
        'res_type': batch['res_type'],
        'token_attention_mask': batch['token_attention_mask'],
        'token_bonds': batch['token_bonds'],
        'ref_pos': batch['ref_pos'],
        'atom_attention_mask': batch['atom_attention_mask'],
        'ref_space_uid': batch['ref_space_uid'],
        'atom_to_token': batch['atom_to_token'],
        'atom_bonds': batch['atom_bonds'],
        'atom_mask': batch['atom_mask'],
        'atom_element': batch['atom_element'],
        'atom_charge': batch['atom_charge'],
        'atom_name_chars': batch['atom_name_chars'],
        'distogram_atom_idx': batch['distogram_atom_idx'],
    }


def write_atom_pdb(path, atom_coords, atom_mask, sequence, confidence=None):
    assert atom_coords.ndim == 3 and atom_coords.size(-1) == 3
    assert atom_mask.shape == atom_coords.shape[:2]
    assert len(sequence) == atom_coords.size(0)

    three_letter = {
        'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS',
        'Q': 'GLN', 'E': 'GLU', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE',
        'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO',
        'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL',
        'X': 'UNK',
    }
    atom_coords = atom_coords.detach().cpu()
    atom_mask = atom_mask.detach().cpu()
    confidence_values = (
        confidence.detach().cpu()
        if confidence is not None
        else torch.zeros(atom_coords.size(0))
    )
    lines = ['REMARK 900 EDUCATIONAL HEAVY-ATOM MODEL; NOT A CHEMICAL STRUCTURE']
    atom_serial = 1
    for residue_index, residue in enumerate(sequence, start=1):
        b_factor = float(confidence_values[residue_index - 1])
        residue_name = three_letter.get(residue, 'UNK')
        for atom_index, atom_name in enumerate(ATOM_SLOTS):
            if not bool(atom_mask[residue_index - 1, atom_index]):
                continue
            xyz = atom_coords[residue_index - 1, atom_index]
            element = atom_name[0] if atom_name[0] in {'C', 'N', 'O', 'S'} else 'C'
            atom_field = f' {atom_name:<3s}' if len(atom_name) < 4 else atom_name[:4]
            lines.append(
                f'ATOM  {atom_serial:5d} {atom_field} {residue_name:>3s} A'
                f'{residue_index:4d}    {float(xyz[0]):8.3f}{float(xyz[1]):8.3f}'
                f'{float(xyz[2]):8.3f}  1.00{b_factor:6.2f}          {element:>2s}'
            )
            atom_serial += 1
    lines.extend(['TER', 'END'])
    Path(path).write_text('\n'.join(lines) + '\n', encoding='ascii')


if __name__ == '__main__':
    torch.manual_seed(1337)
    tiny_config = ProteinFoldingConfig(
        block_size=16,
        lm_dim=32,
        lm_layers=1,
        lm_heads=4,
        single_dim=32,
        pair_dim=16,
        pair_layers=1,
        recycle_loops=2,
        time_dim=16,
        denoiser_layers=1,
        distance_rbf_bins=8,
        distogram_bins=16,
        sampling_steps=4,
    )
    tiny_model = ProteinFolding(tiny_config)
    tiny_batch = make_synthetic_batch(batch_size=2, block_size=16)
    tiny_output = tiny_model(
        **folding_model_inputs(tiny_batch),
        target_atom_coords=tiny_batch['atom_coords'],
    )
    parameter_count = sum(
        parameter.numel() for parameter in tiny_model.parameters()
    )
    print(f'parameters: {parameter_count:,}')
    print(f"loss: {float(tiny_output['loss'].detach()):.4f}")
    print(f"predicted atom coordinates: {tuple(tiny_output['pred_atom_coords'].shape)}")
