from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

from model.protein_tokenizer import (
    ESM_PAD_ID,
    MASK_ID,
    MLM_IGNORE_INDEX,
    ProteinTokenizer,
)


class RotaryPositionEmbedding(nn.Module):
    def __init__(self, head_dim):
        super().__init__()
        assert head_dim % 2 == 0
        rotary_pairs = head_dim // 2
        inverse_frequencies = 1.0 / (
            10000 ** (torch.arange(rotary_pairs, dtype=torch.float32) / rotary_pairs)
        )
        self.register_buffer('inverse_frequencies', inverse_frequencies)
        self.rotary_pairs = rotary_pairs

    def forward(self, query, key):
        sequence_length = query.size(2)
        positions = torch.arange(
            sequence_length,
            device=query.device,
            dtype=self.inverse_frequencies.dtype,
        )
        angles = positions[:, None] * self.inverse_frequencies[None, :]
        cosine = torch.cos(angles).to(query.dtype)[None, None, :, :]
        sine = torch.sin(angles).to(query.dtype)[None, None, :, :]
        return self.apply_rotary(query, cosine, sine), self.apply_rotary(key, cosine, sine)

    def apply_rotary(self, tensor, cosine, sine):
        batch_size, num_heads, sequence_length, head_dim = tensor.shape
        tensor_pairs = tensor.view(
            batch_size,
            num_heads,
            sequence_length,
            self.rotary_pairs,
            2,
        )
        first = tensor_pairs[..., 0]
        second = tensor_pairs[..., 1]
        rotated = torch.stack(
            [
                first * cosine - second * sine,
                first * sine + second * cosine,
            ],
            dim=-1,
        )
        return rotated.view(batch_size, num_heads, sequence_length, head_dim)


class ProteinSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.embed_dim % config.lm_heads == 0
        self.qkv = nn.Linear(config.embed_dim, 3 * config.embed_dim, bias=False)
        self.output = nn.Linear(config.embed_dim, config.embed_dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.num_heads = config.lm_heads
        self.embed_dim = config.embed_dim
        self.head_dim = config.embed_dim // config.lm_heads
        self.query_norm = nn.LayerNorm(self.head_dim)
        self.key_norm = nn.LayerNorm(self.head_dim)
        self.rope = RotaryPositionEmbedding(self.head_dim)

    def forward(self, hidden_states, attention_mask, sequence_id):
        batch_size, sequence_length, embed_dim = hidden_states.shape
        query_key_value = self.qkv(hidden_states)
        query, key, value = query_key_value.split(self.embed_dim, dim=-1)

        query = query.view(batch_size, sequence_length, self.num_heads, self.head_dim)
        key = key.view(batch_size, sequence_length, self.num_heads, self.head_dim)
        value = value.view(batch_size, sequence_length, self.num_heads, self.head_dim)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        query = self.query_norm(query)
        key = self.key_norm(key)
        query, key = self.rope(query, key)

        if sequence_id is None:
            attention_bias_mask = attention_mask[:, None, None, :].bool()
        else:
            valid_sequence = sequence_id >= 0
            attention_bias_mask = (
                sequence_id[:, None, :, None] == sequence_id[:, None, None, :]
            )
            attention_bias_mask = (
                attention_bias_mask
                & valid_sequence[:, None, :, None]
                & valid_sequence[:, None, None, :]
            )
        dropout_p = self.dropout.p if self.training else 0.0
        attended_values = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_bias_mask,
            dropout_p=dropout_p,
        )
        attended_values = attended_values.transpose(1, 2).contiguous().view(
            batch_size,
            sequence_length,
            embed_dim,
        )
        attention_output = self.output(attended_values)
        attention_output = attention_output * attention_mask.to(attention_output.dtype)[
            ...,
            None,
        ]
        return attention_output


class ProteinMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc = nn.Linear(config.embed_dim, 8 * config.embed_dim, bias=False)
        self.projection = nn.Linear(4 * config.embed_dim, config.embed_dim, bias=False)

    def forward(self, hidden_states):
        hidden_states = self.fc(hidden_states)
        gate, values = hidden_states.chunk(2, dim=-1)
        hidden_states = F.silu(gate) * values
        hidden_states = self.projection(hidden_states)
        return hidden_states


class ProteinTransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.embed_dim)
        self.attention = ProteinSelfAttention(config)
        self.mlp_norm = nn.LayerNorm(config.embed_dim)
        self.mlp = ProteinMLP(config)

    def forward(self, hidden_states, attention_mask, sequence_id=None):
        attention_output = self.attention(
            self.attention_norm(hidden_states),
            attention_mask,
            sequence_id,
        )
        hidden_states = hidden_states + attention_output
        hidden_states = hidden_states + self.mlp(self.mlp_norm(hidden_states))
        hidden_states = hidden_states * attention_mask.to(hidden_states.dtype)[..., None]
        return hidden_states


@dataclass
class ProteinLMConfig:
    vocab_size: int = 64
    context_size: int = 64
    embed_dim: int = 64
    lm_heads: int = 4
    lm_layers: int = 3
    mlm_mask_probability: float = 0.15
    dropout: float = 0.10


class ProteinLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.lm_layers > 0
        self.config = config
        self.tokenizer = ProteinTokenizer()
        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.embed_dim,
            padding_idx=ESM_PAD_ID,
        )
        self.blocks = nn.ModuleList(
            [ProteinTransformerBlock(config) for _ in range(config.lm_layers)]
        )
        self.final_norm = nn.LayerNorm(config.embed_dim)
        self.sequence_head = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim),
            nn.GELU(),
            nn.LayerNorm(config.embed_dim),
            nn.Linear(config.embed_dim, config.vocab_size),
        )
        self.dropout = nn.Dropout(config.dropout)
        self.apply(self._init_weights)
        with torch.no_grad():
            self.token_embedding.weight[ESM_PAD_ID].zero_()

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, input_ids, residue_mask, mlm_targets=None, asym_id=None, residue_index=None, mol_type=None):
        assert input_ids.size(1) <= self.config.context_size
        protein_mask = residue_mask.bool()
        if mol_type is not None:
            protein_mask = protein_mask & (mol_type == 0)

        lm_chain_count = 1
        if asym_id is not None:
            for batch_index in range(input_ids.size(0)):
                valid_chains = asym_id[batch_index, protein_mask[batch_index]]
                if int(valid_chains.numel()) > 0:
                    lm_chain_count = max(lm_chain_count, int(valid_chains.unique().numel()))

        lm_input_ids, lm_attention_mask, sequence_id, residue_to_lm_index = (
            self.tokenizer.wrap_for_lm(
                input_ids,
                residue_mask,
                asym_id=asym_id,
                residue_index=residue_index,
                mol_type=mol_type,
                lm_length=self.config.context_size + 2 * lm_chain_count,
            )
        )
        token_embeddings = self.token_embedding(lm_input_ids)
        hidden_states = self.dropout(token_embeddings)
        hidden_states = hidden_states * lm_attention_mask.to(hidden_states.dtype)[
            ...,
            None,
        ]

        hidden_state_stack = [hidden_states]
        for block in self.blocks:
            hidden_states = block(hidden_states, lm_attention_mask, sequence_id)
            hidden_state_stack.append(hidden_states)

        final_lm_embedding = self.final_norm(hidden_states)
        final_lm_embedding = final_lm_embedding * lm_attention_mask.to(
            final_lm_embedding.dtype
        )[..., None]
        hidden_state_stack[-1] = final_lm_embedding

        residue_gather_index = residue_to_lm_index.clamp_min(0).unsqueeze(-1).expand(
            -1,
            -1,
            self.config.embed_dim,
        )
        final_residue_states = final_lm_embedding.gather(dim=1, index=residue_gather_index)
        final_residue_states = final_residue_states * protein_mask.to(final_residue_states.dtype)[..., None]

        hidden_state_stack = torch.stack(hidden_state_stack, dim=2)
        layer_gather_index = residue_gather_index.unsqueeze(2).expand(
            -1,
            -1,
            hidden_state_stack.size(2),
            -1,
        )
        hidden_states = hidden_state_stack.gather(dim=1, index=layer_gather_index)
        hidden_states = hidden_states * protein_mask.to(hidden_states.dtype)[..., None, None]

        mlm_logits = self.sequence_head(final_residue_states)
        mlm_logits = mlm_logits * protein_mask.to(mlm_logits.dtype)[..., None]

        mlm_loss = None
        if mlm_targets is not None:
            mlm_targets = torch.where(
                protein_mask,
                mlm_targets,
                torch.full_like(mlm_targets, MLM_IGNORE_INDEX),
            )
            if (mlm_targets != MLM_IGNORE_INDEX).any():
                mlm_loss = F.cross_entropy(
                    mlm_logits.reshape(-1, mlm_logits.size(-1)),
                    mlm_targets.reshape(-1),
                    ignore_index=MLM_IGNORE_INDEX,
                )
            else:
                mlm_loss = mlm_logits.sum() * 0.0

        return {
            'hidden_states': hidden_states,
            'mlm_logits': mlm_logits,
            'loss': mlm_loss,
            'lm_input_ids': lm_input_ids,
            'lm_attention_mask': lm_attention_mask,
            'sequence_id': sequence_id,
            'residue_to_lm_index': residue_to_lm_index,
        }
