"""
Download the HellaSwag validation set (10,042 examples) used for evaluation.

Source: https://huggingface.co/datasets/Rowan/hellaswag
Each row has a context `ctx`, 4 candidate `endings`, and the index of the correct one in `label` (stored as a string).
Tokenization happens at eval time in GPT2_FromScratch/eval.py.

Usage:
    python prepare_hellaswag.py
    python prepare_hellaswag.py --out_dir /content/hellaswag  # e.g. on Colab
"""

import argparse
import os

import pandas as pd
from huggingface_hub import hf_hub_download

REPO_ID = "Rowan/hellaswag"
FILENAME = "data/validation-00000-of-00001.parquet"
NUM_EXAMPLES = 10042


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--out_dir",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "hellaswag"),
        help="where the parquet file is stored",
    )
    args = parser.parse_args()

    path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME, repo_type="dataset", local_dir=args.out_dir)
    df = pd.read_parquet(path)
    assert len(df) == NUM_EXAMPLES, f"expected {NUM_EXAMPLES} examples, got {len(df)}"
    assert all(len(e) == 4 for e in df["endings"]), "every example must have 4 endings"
    print(f"ok {path}: {len(df):,} examples")


if __name__ == "__main__":
    main()
