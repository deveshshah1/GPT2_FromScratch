"""
Download pre-tokenized FineWeb-Edu shards (GPT-2 tokenizer) from Hugging Face.

Source: https://huggingface.co/datasets/kjj0/finewebedu10B-gpt2 (used by llm.c / modded-nanogpt)
  - finewebedu_val_000000.bin           100M tokens
  - finewebedu_train_000001..099.bin    100M tokens each (~10B total)

Shard format: 256 int32 header (magic 20240520, version 1, num_tokens, ...), then num_tokens uint16 tokens.

Usage:
    python prepare_data.py                      # 10 train shards = 1B tokens, plus the val shard
    python prepare_data.py --num_train_shards 2 # 200M tokens
    python prepare_data.py --out_dir /content/finewebedu  # e.g. fast local disk on Colab
"""

import argparse
import os

import numpy as np
from huggingface_hub import hf_hub_download

REPO_ID = "kjj0/finewebedu10B-gpt2"
TOKENS_PER_SHARD = 100_000_000
HEADER_INTS = 256
MAGIC = 20240520
VERSION = 1


def shard_name(split, idx):
    return f"finewebedu_{split}_{idx:06d}.bin"


def verify_shard(path):
    # Check the header and that the file holds exactly num_tokens uint16 tokens
    header = np.fromfile(path, dtype=np.int32, count=HEADER_INTS)
    assert header[0] == MAGIC, f"{path}: bad magic {header[0]}"
    assert header[1] == VERSION, f"{path}: unsupported version {header[1]}"
    num_tokens = int(header[2])
    expected_bytes = HEADER_INTS * 4 + num_tokens * 2
    actual_bytes = os.path.getsize(path)
    assert actual_bytes == expected_bytes, f"{path}: {actual_bytes} bytes, expected {expected_bytes}"
    return num_tokens


def download_shard(fname, out_dir):
    path = os.path.join(out_dir, fname)
    if not os.path.exists(path):
        print(f"downloading {fname}")
        hf_hub_download(repo_id=REPO_ID, filename=fname, repo_type="dataset", local_dir=out_dir)
    num_tokens = verify_shard(path)
    print(f"ok {fname}: {num_tokens:,} tokens")
    return num_tokens


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--num_train_shards", type=int, default=10, help="100M tokens per shard (max 99)")
    parser.add_argument(
        "--out_dir",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "finewebedu"),
        help="where the shards are stored",
    )
    args = parser.parse_args()
    assert 1 <= args.num_train_shards <= 99, "num_train_shards must be in [1, 99]"
    os.makedirs(args.out_dir, exist_ok=True)

    val_tokens = download_shard(shard_name("val", 0), args.out_dir)
    train_tokens = sum(
        download_shard(shard_name("train", i), args.out_dir) for i in range(1, args.num_train_shards + 1)
    )
    print(f"\nshards in {args.out_dir}")
    print(f"train: {train_tokens:,} tokens ({args.num_train_shards} shards)")
    print(f"val:   {val_tokens:,} tokens")


if __name__ == "__main__":
    main()
