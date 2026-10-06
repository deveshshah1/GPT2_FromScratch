import os

import tiktoken
import torch
import yaml

CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "configs", "config_training_small.yaml"
)

with open(CONFIG_PATH, "r") as f:
    config_training = yaml.safe_load(f)
    config_training = {k: v["value"] for k, v in config_training.items()}


class TextDataset(torch.utils.data.Dataset):
    """
    Next-token prediction dataset over a single text file, tokenized with the
    GPT-2 BPE tokenizer. The token stream is split into train/val by position,
    then cut into non-overlapping sequences of length sequence_length. Each item
    is (x, y) where y is x shifted one token to the right.
    """

    def __init__(
        self,
        dataset_path: str = "./dataset/input.txt",
        split: str = "train",
        sequence_length: int = 1024,
        train_split: float = 0.9,
        **kwargs,
    ):
        super().__init__()
        assert split in {"train", "val"}, f"unknown split: {split}"
        self.split = split
        self.sequence_length = sequence_length

        with open(dataset_path, "r", encoding="utf-8") as f:
            text = f.read()
        self.enc = self.get_tokenizer()
        tokens = torch.tensor(self.enc.encode(text), dtype=torch.long)

        n = int(train_split * len(tokens))
        self.tokens = tokens[:n] if split == "train" else tokens[n:]

        # -1 so the final target token of the last sequence is in range
        self.num_sequences = (len(self.tokens) - 1) // sequence_length

        print(f"{split} split: {len(self.tokens):,} tokens -> {self.num_sequences:,} sequences of {sequence_length}")

    @staticmethod
    def get_tokenizer():
        return tiktoken.get_encoding("gpt2")

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        start = idx * self.sequence_length
        buf = self.tokens[start : start + self.sequence_length + 1]
        x = buf[:-1]  # inputs
        y = buf[1:]  # targets
        return x, y


if __name__ == "__main__":
    dataset_configs = config_training["dataset_configs"]

    for split in ["train", "val"]:
        dataset = TextDataset(split=split, **dataset_configs)
        print(
            f"{split:5s}: {len(dataset.tokens):,} tokens -> "
            f"{len(dataset):,} sequences of {dataset.sequence_length}"
        )

    x, y = dataset[0]
    print(f"x shape: {tuple(x.shape)}, y shape: {tuple(y.shape)}")
    assert torch.equal(x[1:], y[:-1]), "targets must be inputs shifted by one"
    print(f"x[:16] decoded: {dataset.enc.decode(x[:16].tolist())!r}")
    print(f"y[:16] decoded: {dataset.enc.decode(y[:16].tolist())!r}")
    print("\nDataset loaded successfully.")
