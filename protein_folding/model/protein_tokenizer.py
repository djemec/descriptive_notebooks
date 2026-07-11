import torch


# These are the ESM-C IDs consumed by ESMFold2 for ordinary protein residues.
BOS_ID = 0
ESM_PAD_ID = 1
EOS_ID = 2
UNKNOWN_ID = 3
NON_PROTEIN_ID = 24
MASK_ID = 32
RES_PAD_ID = 0
MLM_IGNORE_INDEX = -100

ESM_PROTEIN_VOCAB = {
    'L': 4,
    'A': 5,
    'G': 6,
    'V': 7,
    'S': 8,
    'E': 9,
    'R': 10,
    'T': 11,
    'I': 12,
    'D': 13,
    'P': 14,
    'K': 15,
    'Q': 16,
    'N': 17,
    'F': 18,
    'Y': 19,
    'M': 20,
    'H': 21,
    'W': 22,
    'C': 23,
    'X': UNKNOWN_ID,
}

INPUT_ID_TO_TOKEN = {
    BOS_ID: '<bos>',
    ESM_PAD_ID: '<pad>',
    EOS_ID: '<eos>',
    UNKNOWN_ID: 'X',
    NON_PROTEIN_ID: '<non_protein>',
    MASK_ID: '<mask>',
}
INPUT_ID_TO_TOKEN.update(
    {token_id: token for token, token_id in ESM_PROTEIN_VOCAB.items()}
)

# ESMFold2 uses a second residue encoding for structural features.
RES_TYPE_VOCAB = {
    'A': 2,
    'R': 3,
    'N': 4,
    'D': 5,
    'C': 6,
    'Q': 7,
    'E': 8,
    'G': 9,
    'H': 10,
    'I': 11,
    'L': 12,
    'K': 13,
    'M': 14,
    'F': 15,
    'P': 16,
    'S': 17,
    'T': 18,
    'W': 19,
    'Y': 20,
    'V': 21,
    'X': 22,
}

PDB_THREE_TO_ONE = {
    'ALA': 'A',
    'ARG': 'R',
    'ASN': 'N',
    'ASP': 'D',
    'CYS': 'C',
    'GLN': 'Q',
    'GLU': 'E',
    'GLY': 'G',
    'HIS': 'H',
    'ILE': 'I',
    'LEU': 'L',
    'LYS': 'K',
    'MET': 'M',
    'PHE': 'F',
    'PRO': 'P',
    'SER': 'S',
    'THR': 'T',
    'TRP': 'W',
    'TYR': 'Y',
    'VAL': 'V',
}

PDB_MODIFIED_THREE_TO_ONE = {
    'MSE': 'M',
}


class ProteinTokenizer:

    def encode(self, sequence):
        sequence = sequence.strip().upper()
        input_ids = []
        res_type = []
        for residue in sequence:
            normalized = residue if residue in ESM_PROTEIN_VOCAB else 'X'
            input_ids.append(ESM_PROTEIN_VOCAB[normalized])
            res_type.append(RES_TYPE_VOCAB.get(normalized, RES_TYPE_VOCAB['X']))
        return input_ids, res_type

    def pdb_residue_to_one_letter(self, residue_name):
        residue_name = residue_name.strip().upper()
        if residue_name in PDB_THREE_TO_ONE:
            return PDB_THREE_TO_ONE[residue_name]
        if residue_name in PDB_MODIFIED_THREE_TO_ONE:
            return PDB_MODIFIED_THREE_TO_ONE[residue_name]
        return None

    def encode_pdb_residue_names(self, residue_names):
        sequence = []
        unsupported = []
        for residue_name in residue_names:
            one_letter = self.pdb_residue_to_one_letter(residue_name)
            if one_letter is None:
                unsupported.append(str(residue_name).strip().upper())
            else:
                sequence.append(one_letter)
        if unsupported:
            return None, None, unsupported
        input_ids, res_type = self.encode(''.join(sequence))
        return input_ids, res_type, unsupported

    def decode(self, input_ids, skip_special=True):
        residues = []
        for token_id in input_ids:
            token = INPUT_ID_TO_TOKEN.get(int(token_id), 'X')
            if token.startswith('<'):
                if not skip_special:
                    residues.append(token)
                continue
            residues.append(token)
        return ''.join(residues)

    def wrap_for_lm(self, input_ids, residue_mask, asym_id=None, residue_index=None, mol_type=None, lm_length=None):
        assert input_ids.ndim == 2
        assert input_ids.shape == residue_mask.shape
        residue_mask = residue_mask.bool()
        if asym_id is None:
            asym_id = torch.zeros_like(input_ids)
        if residue_index is None:
            residue_index = torch.arange(
                input_ids.size(1),
                device=input_ids.device,
                dtype=torch.long,
            )[None, :].expand_as(input_ids)
        if mol_type is None:
            mol_type = torch.zeros_like(input_ids)
        assert asym_id.shape == input_ids.shape
        assert residue_index.shape == input_ids.shape
        assert mol_type.shape == input_ids.shape
        batch_size, residue_length = input_ids.shape
        required_lm_length = 0
        protein_mask = residue_mask & (mol_type == 0)
        for batch_index in range(batch_size):
            residue_positions = torch.nonzero(
                protein_mask[batch_index], as_tuple=False
            ).squeeze(-1)
            if int(residue_positions.numel()) == 0:
                required_lm_length = max(required_lm_length, 2)
                continue
            chain_count = int(asym_id[batch_index, residue_positions].unique().numel())
            required_lm_length = max(
                required_lm_length,
                int(residue_positions.numel()) + 2 * chain_count,
            )
        if lm_length is None:
            lm_length = required_lm_length
        assert lm_length >= required_lm_length

        lm_input_ids = torch.full(
            (batch_size, lm_length),
            ESM_PAD_ID,
            dtype=torch.long,
            device=input_ids.device,
        )
        lm_attention_mask = torch.zeros(
            (batch_size, lm_length),
            dtype=torch.bool,
            device=input_ids.device,
        )
        sequence_id = torch.full(
            (batch_size, lm_length),
            -1,
            dtype=torch.long,
            device=input_ids.device,
        )
        residue_to_lm_index = torch.full(
            (batch_size, residue_length),
            -1,
            dtype=torch.long,
            device=input_ids.device,
        )

        for batch_index in range(batch_size):
            residue_positions = torch.nonzero(
                protein_mask[batch_index], as_tuple=False
            ).squeeze(-1)
            if int(residue_positions.numel()) == 0:
                lm_input_ids[batch_index, 0] = BOS_ID
                lm_input_ids[batch_index, 1] = EOS_ID
                lm_attention_mask[batch_index, :2] = True
                sequence_id[batch_index, :2] = 0
                continue

            keys = torch.stack(
                [
                    asym_id[batch_index, residue_positions],
                    residue_index[batch_index, residue_positions],
                ],
                dim=1,
            )
            unique_keys, inverse = torch.unique(keys, dim=0, return_inverse=True)
            token_positions = torch.arange(
                keys.size(0),
                device=input_ids.device,
                dtype=torch.long,
            )
            first_positions = torch.full(
                (unique_keys.size(0),),
                keys.size(0),
                device=input_ids.device,
                dtype=torch.long,
            )
            first_positions.scatter_reduce_(
                0,
                inverse,
                token_positions,
                reduce='amin',
                include_self=True,
            )
            ordered_unique = torch.argsort(first_positions)
            ordered_first_positions = first_positions[ordered_unique]
            ordered_input_ids = input_ids[batch_index, residue_positions][
                ordered_first_positions
            ]
            ordered_asym_id = asym_id[batch_index, residue_positions][
                ordered_first_positions
            ]
            unique_to_ordered = torch.empty_like(ordered_unique)
            unique_to_ordered[ordered_unique] = torch.arange(
                ordered_unique.numel(),
                device=input_ids.device,
                dtype=torch.long,
            )
            residue_to_ordered = unique_to_ordered[inverse]

            cursor = 0
            ordered_to_lm_index = torch.empty(
                ordered_input_ids.size(0),
                device=input_ids.device,
                dtype=torch.long,
            )
            chain_ids = ordered_asym_id.unique(sorted=True)
            for chain_number, chain_id in enumerate(chain_ids):
                lm_input_ids[batch_index, cursor] = BOS_ID
                sequence_id[batch_index, cursor] = int(chain_number)
                cursor += 1
                chain_positions = torch.nonzero(
                    ordered_asym_id == chain_id, as_tuple=False
                ).squeeze(-1)
                chain_length = int(chain_positions.numel())
                lm_input_ids[batch_index, cursor : cursor + chain_length] = (
                    ordered_input_ids[chain_positions]
                )
                sequence_id[batch_index, cursor : cursor + chain_length] = int(chain_number)
                ordered_to_lm_index[chain_positions] = torch.arange(
                    cursor,
                    cursor + chain_length,
                    device=input_ids.device,
                    dtype=torch.long,
                )
                cursor += chain_length
                lm_input_ids[batch_index, cursor] = EOS_ID
                sequence_id[batch_index, cursor] = int(chain_number)
                cursor += 1

            lm_attention_mask[batch_index, :cursor] = True
            residue_to_lm_index[batch_index, residue_positions] = (
                ordered_to_lm_index[residue_to_ordered]
            )

        return lm_input_ids, lm_attention_mask, sequence_id, residue_to_lm_index

    def mask_tokens(self, input_ids, residue_mask, mask_probability=0.15, generator=None):
        assert 0.0 <= mask_probability <= 1.0
        residue_mask = residue_mask.bool()
        random_values = torch.rand(
            input_ids.shape,
            device=input_ids.device,
            generator=generator,
        )
        masked_positions = (random_values < mask_probability) & residue_mask
        masked_input_ids = input_ids.clone()
        masked_input_ids[masked_positions] = MASK_ID
        mlm_targets = torch.full_like(input_ids, MLM_IGNORE_INDEX)
        mlm_targets[masked_positions] = input_ids[masked_positions]
        return masked_input_ids, mlm_targets


def make_sequence_batch(sequences, block_size=64, device=None):
    assert len(sequences) > 0
    tokenizer = ProteinTokenizer()
    input_ids = torch.full(
        (len(sequences), block_size),
        ESM_PAD_ID,
        dtype=torch.long,
    )
    res_type = torch.full(
        (len(sequences), block_size),
        RES_PAD_ID,
        dtype=torch.long,
    )
    residue_mask = torch.zeros(len(sequences), block_size, dtype=torch.bool)

    for batch_index, sequence in enumerate(sequences):
        sequence_input_ids, sequence_res_types = tokenizer.encode(sequence)
        assert len(sequence_input_ids) > 0
        assert len(sequence_input_ids) <= block_size
        length = len(sequence_input_ids)
        input_ids[batch_index, :length] = torch.tensor(sequence_input_ids)
        res_type[batch_index, :length] = torch.tensor(sequence_res_types)
        residue_mask[batch_index, :length] = True

    batch = {
        'input_ids': input_ids,
        'res_type': res_type,
        'residue_mask': residue_mask,
        'coord_mask': residue_mask.clone(),
    }
    if device is not None:
        batch = {key: value.to(device) for key, value in batch.items()}
    return batch
