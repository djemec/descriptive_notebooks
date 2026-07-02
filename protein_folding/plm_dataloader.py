from pathlib import Path
import random

import numpy as np
import torch


class ProteinBatchLoader:
    def __init__(self, data_dir, split, batch_size, device, seed=1337):
        self.data_dir = Path(data_dir)
        self.split = split
        self.batch_size = batch_size
        self.device = device
        self.random = random.Random(seed)
        self.numpy_random = np.random.RandomState(seed)
        self.shards = sorted((self.data_dir / split).glob('*.npz'))
        if not self.shards:
            raise FileNotFoundError(f'no .npz shards found in {self.data_dir / split}')
        self.num_examples = self._count_examples()
        self.reset()

    def _count_examples(self):
        total = 0
        for shard_path in self.shards:
            with np.load(shard_path, allow_pickle=False) as shard:
                total += len(shard['input_ids'])
        return total

    def reset(self):
        self.shard_order = list(self.shards)
        if self.split == 'train':
            self.random.shuffle(self.shard_order)
        self.shard_index = 0
        self.position = 0
        self._load_current_shard()

    def _load_current_shard(self):
        with np.load(self.shard_order[self.shard_index], allow_pickle=False) as shard:
            self.input_ids = shard['input_ids'].astype(np.int64)
            self.residue_mask = shard['residue_mask'].astype(np.bool_)
        self.order = self.numpy_random.permutation(len(self.input_ids))
        if self.split != 'train':
            self.order = np.arange(len(self.input_ids))
        self.position = 0

    def _advance_shard(self):
        self.shard_index += 1
        if self.shard_index >= len(self.shard_order):
            self.reset()
        else:
            self._load_current_shard()

    def _take_examples(self, count):
        input_pieces = []
        mask_pieces = []
        while count > 0:
            remaining = len(self.order) - self.position
            take = min(count, remaining)
            indices = self.order[self.position:self.position + take]
            input_pieces.append(self.input_ids[indices])
            mask_pieces.append(self.residue_mask[indices])
            self.position += take
            count -= take
            if self.position >= len(self.order):
                self._advance_shard()
        return np.concatenate(input_pieces), np.concatenate(mask_pieces)

    def next_batch(self):
        input_ids_array, residue_mask_array = self._take_examples(self.batch_size)
        input_ids = torch.tensor(input_ids_array, dtype=torch.long, device=self.device)
        residue_mask = torch.tensor(residue_mask_array, dtype=torch.bool, device=self.device)
        return {
            'input_ids': input_ids,
            'residue_mask': residue_mask,
        }
