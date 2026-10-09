import json
import os
import time

import pandas as pd
import torch
from custom_dataset import ShardDataset
from pyL_modules import PyLModel
from torch.nn import functional as F

HELLASWAG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "dataset", "hellaswag", "data", "validation-00000-of-00001.parquet",
)


def render_example(enc, ctx, endings):
    """
    Build the 4 candidate sequences (context + " " + ending) for one HellaSwag example, right-padded
    to the same length. mask is 1 on the ending tokens, which are the only ones that get scored.
    """
    ctx_tokens = enc.encode(ctx)
    rows = [ctx_tokens + enc.encode(" " + ending) for ending in endings]
    T = max(len(row) for row in rows)
    tokens = torch.zeros((4, T), dtype=torch.long)
    mask = torch.zeros((4, T), dtype=torch.long)
    for i, row in enumerate(rows):
        tokens[i, : len(row)] = torch.tensor(row)
        mask[i, len(ctx_tokens) : len(row)] = 1
    return tokens, mask


@torch.no_grad()
def eval_hellaswag(net, device, path=HELLASWAG_PATH):
    """
    The model picks the ending it finds most likely. acc uses the total loss over the ending tokens,
    acc_norm the average loss per ending token (doesn't favour short endings).
    """
    df = pd.read_parquet(path)
    enc = ShardDataset.get_tokenizer()
    num_correct, num_correct_norm = 0, 0
    for row in df.itertuples():
        tokens, mask = render_example(enc, row.ctx, row.endings)
        tokens, mask = tokens.to(device), mask.to(device)
        logits = net(tokens)  # (4, T, vocab_size)
        # loss of predicting token t+1 from position t
        losses = F.cross_entropy(
            logits[:, :-1].flatten(0, 1), tokens[:, 1:].flatten(), reduction="none"
        ).view(4, -1)
        shift_mask = mask[:, 1:]
        sum_loss = (losses * shift_mask).sum(dim=1)
        avg_loss = sum_loss / shift_mask.sum(dim=1)
        label = int(row.label)
        num_correct += int(sum_loss.argmin().item() == label)
        num_correct_norm += int(avg_loss.argmin().item() == label)
    return num_correct / len(df), num_correct_norm / len(df)


def evaluate(ckpt_path):
    print(f"------------------Evaluating {ckpt_path}------------------")

    # Results go next to the checkpoints dir: <model_dir>/<experiment>/predictions/
    exp_dir = os.path.dirname(os.path.dirname(os.path.abspath(ckpt_path)))
    results_dir = os.path.join(exp_dir, "predictions")
    os.makedirs(results_dir, exist_ok=True)

    # Load the model
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    global_step = torch.load(ckpt_path, map_location="cpu", weights_only=False)["global_step"]
    model = PyLModel.load_from_checkpoint(ckpt_path, map_location=device, wandb_logger=None)
    model.eval()
    net = model.model.forward

    t0 = time.time()
    acc, acc_norm = eval_hellaswag(net, device)
    print(f"hellaswag: acc {acc:.4f}, acc_norm {acc_norm:.4f} ({time.time() - t0:.0f}s)")

    results = {
        "checkpoint": ckpt_path,
        "global_step": global_step,
        "hellaswag_acc": acc,
        "hellaswag_acc_norm": acc_norm,
    }
    ckpt_name = os.path.splitext(os.path.basename(ckpt_path))[0]
    out_path = os.path.join(results_dir, f"{ckpt_name}_eval.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {out_path}")
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt_paths", nargs="+", help="one or more checkpoints to evaluate")
    args = parser.parse_args()

    all_results = [evaluate(ckpt_path) for ckpt_path in args.ckpt_paths]

    print(f"\n{'checkpoint':<70} {'step':>6} {'hs_acc':>7} {'hs_acc_norm':>12}")
    for r in all_results:
        print(
            f"{r['checkpoint']:<70} {r['global_step']:>6} "
            f"{r['hellaswag_acc']:>7.4f} {r['hellaswag_acc_norm']:>12.4f}"
        )
