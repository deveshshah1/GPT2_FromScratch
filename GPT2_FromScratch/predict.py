import os
import torch
from pyL_modules import PyLModel


def predict(ckpt_path, num_samples=10, max_length=64):
    print(f"------------------Generating samples for {ckpt_path}------------------")

    # Results go next to the checkpoints dir: <model_dir>/<experiment>/predictions/
    exp_dir = os.path.dirname(os.path.dirname(os.path.abspath(ckpt_path)))
    results_dir = os.path.join(exp_dir, "predictions")
    os.makedirs(results_dir, exist_ok=True)

    # Load the model
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = PyLModel.load_from_checkpoint(ckpt_path, map_location=device, wandb_logger=None)
    model.eval()

    # Sample with a fixed seed so different checkpoints are compared on the same random draws
    samples = model.generate_samples(num_samples=num_samples, max_length=max_length)

    ckpt_name = os.path.splitext(os.path.basename(ckpt_path))[0]
    out_path = os.path.join(results_dir, f"{ckpt_name}_samples.txt")
    with open(out_path, "w") as f:
        f.write(f"checkpoint: {ckpt_path}\nglobal_step: {ckpt['global_step']}\n\n")
        for i, sample in enumerate(samples):
            print(f"sample {i}: {sample}\n")
            f.write(f"sample {i}: {sample}\n\n")
    print(f"Saved samples to {out_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt_paths", nargs="+", help="one or more checkpoints to sample from")
    parser.add_argument("--num_samples", type=int, default=10, help="samples per checkpoint")
    parser.add_argument("--max_length", type=int, default=64, help="total tokens per sample, including the prompt")
    args = parser.parse_args()

    for ckpt_path in args.ckpt_paths:
        predict(ckpt_path, num_samples=args.num_samples, max_length=args.max_length)
