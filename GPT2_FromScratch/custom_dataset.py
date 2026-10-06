import os

import numpy as np
import tiktoken
import torch
import yaml

CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "configs", "config_training_small.yaml"
)

with open(CONFIG_PATH, "r") as f:
    config_training = yaml.safe_load(f)
    config_training = {k: v["value"] for k, v in config_training.items()}

HEADER_BYTES = 256 * 4  # 256 int32 header at the start of every shard


class ShardDataset(torch.utils.data.Dataset):
    """
    Next-token prediction dataset over pre-tokenized GPT-2 shards (see dataset/prepare_data.py).
    Each shard is memory-mapped and cut into non-overlapping sequences of length
    sequence_length. Each item is (x, y) where y is x shifted one token to the right.
    """

    def __init__(
        self,
        dataset_path: str = "./dataset/finewebedu",
        split: str = "train",
        sequence_length: int = 1024,
        **kwargs,
    ):
        super().__init__()
        assert split in {"train", "val"}, f"unknown split: {split}"
        self.sequence_length = sequence_length

        shard_paths = sorted(
            os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f"_{split}_" in f
        )
        assert len(shard_paths) > 0, f"no {split} shards found in {dataset_path}"
        self.shard_paths = shard_paths
        # Memmaps are opened lazily in each DataLoader worker (pickling an open memmap copies its data)
        self.shards = [None] * len(shard_paths)

        # All shards hold the same number of tokens (uint16 = 2 bytes each)
        tokens_per_shard = (os.path.getsize(shard_paths[0]) - HEADER_BYTES) // 2
        assert all(os.path.getsize(p) == os.path.getsize(shard_paths[0]) for p in shard_paths), "shards must be equal size"
        # -1 so the final target token of the last sequence is in range
        self.seqs_per_shard = (tokens_per_shard - 1) // sequence_length
        self.num_sequences = self.seqs_per_shard * len(shard_paths)

        num_tokens = tokens_per_shard * len(shard_paths)
        print(
            f"{split} split: {len(shard_paths)} shards, {num_tokens:,} tokens -> "
            f"{self.num_sequences:,} sequences of {sequence_length}"
        )

    @staticmethod
    def get_tokenizer():
        return tiktoken.get_encoding("gpt2")

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        shard_idx, seq_idx = divmod(idx, self.seqs_per_shard)
        if self.shards[shard_idx] is None:
            self.shards[shard_idx] = np.memmap(
                self.shard_paths[shard_idx], dtype=np.uint16, mode="r", offset=HEADER_BYTES
            )
        start = seq_idx * self.sequence_length
        buf = self.shards[shard_idx][start : start + self.sequence_length + 1]
        buf = torch.from_numpy(buf.astype(np.int64))
        x = buf[:-1]  # inputs
        y = buf[1:]  # targets
        return x, y


if __name__ == "__main__":
    dataset_configs = config_training["dataset_configs"]

    for split in ["train", "val"]:
        dataset = ShardDataset(split=split, **dataset_configs)

    enc = dataset.get_tokenizer()
    x, y = dataset[0]
    print(f"x shape: {tuple(x.shape)}, y shape: {tuple(y.shape)}")
    assert torch.equal(x[1:], y[:-1]), "targets must be inputs shifted by one"
    print(f"x[:16] decoded: {enc.decode(x[:16].tolist())!r}")
    print(f"y[:16] decoded: {enc.decode(y[:16].tolist())!r}")
    print("\nDataset loaded successfully.")
