# Karpathy nanoGPT Tutorials

Reimplementations of the GPT models from Andrej Karpathy's [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) lecture series, written as a learning exercise. Each tutorial is in its own directory.

| Tutorial | Lecture | Directory | Model |
| --- | --- | --- | --- |
| 1 | [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY) | [`tutorial1/`](tutorial1/) | Character-level Transformer trained on Tiny Shakespeare |
| 2 | [Let's reproduce GPT-2 (124M)](https://www.youtube.com/watch?v=l8pRSuU81PU) | [`tutorial2/`](tutorial2/) | GPT-2 (124M) trained on FineWeb-Edu |

## Tutorial 1: Building GPT from scratch

Builds a decoder-only Transformer one piece at a time, starting from a bigram model, and trains it to generate Shakespeare-like text one character at a time.

### Files

- [`download_data.ipynb`](tutorial1/download_data.ipynb): step-by-step notebook. It downloads the Tiny Shakespeare dataset, builds a character-level tokenizer, trains a baseline bigram model, and works up to self-attention through the "mathematical trick" (averaging over past tokens with loops, then matrix multiplication, then softmax, then a single attention head).
- [`train.py`](tutorial1/train.py): the finished training script.

Both tutorials read the Tiny Shakespeare dataset (~1.1M characters) from the shared [`dataset/input.txt`](../dataset/input.txt) at the repo root.

### Model

| Hyperparameter | Value |
| --- | --- |
| Layers | 6 |
| Attention heads | 6 |
| Embedding dimension | 384 |
| Context length | 256 characters |
| Dropout | 0.2 |
| Batch size | 64 |
| Learning rate | 3e-4 (AdamW) |
| Training iterations | 5,000 |

Components: token and position embeddings, multi-head causal self-attention, a feed-forward network with ReLU, pre-norm LayerNorm, and residual connections. The data is split 90/10 into train and validation sets.

### Running

```bash
cd tutorial1
python train.py
```

The script prints train and validation loss every 500 steps, then generates 500 characters of text. The device is hard-coded to `mps` (Apple Silicon). Change `device` at the top of `train.py` to run on `cuda` or `cpu`.

## Tutorial 2: Reproducing GPT-2 (124M)

Reproduces the 124M-parameter GPT-2 model. The architecture matches OpenAI's GPT-2, so the model can either load the pretrained Hugging Face weights or be trained from scratch on the FineWeb-Edu dataset.

### Files

- [`train_gpt2.py`](tutorial2/train_gpt2.py): model definition and training script.
- [`playground.ipynb`](tutorial2/playground.ipynb): scratch notebook for tokenizing with `tiktoken` and building `(x, y)` batches.

### Model

The configuration is the GPT-2 small configuration: 12 layers, 12 heads, an embedding dimension of 768, and a context length of 1024 tokens. The vocabulary size is padded from 50,257 to 50,304 so that it is a multiple of 64, which makes GPU kernels faster.

Architecture details:

- Fused QKV projection, with `F.scaled_dot_product_attention` (Flash Attention)
- GELU (tanh approximation) in the MLP
- The token embedding and output head share weights
- Residual projections are initialized with weights scaled by `1/sqrt(2 * n_layer)`
- `GPT.from_pretrained(...)` loads the `gpt2`, `gpt2-medium`, `gpt2-large` or `gpt2-xl` weights from Hugging Face

### Training setup

| Setting | Value |
| --- | --- |
| Total batch size | 524,288 tokens (2^19), reached with gradient accumulation |
| Micro batch | 64 × 1024 tokens |
| Learning rate | Linear warmup for 715 steps to 6e-4, then cosine decay to 6e-5 |
| Max steps | 19,073 (~1 epoch of 10B tokens) |
| Optimizer | AdamW (β = 0.9, 0.95), weight decay 0.1 on 2D parameters only, fused on CUDA |
| Gradient clipping | 1.0 |
| Precision | bfloat16 autocast, TF32 matmuls |
| Distributed | DDP through `torchrun` |

Every 250 steps, the script measures validation loss, measures HellaSwag accuracy, and samples text from the prompt `"Hello, I'm a language model,"`. Logs go to `log/log.txt`, and a checkpoint is saved to `log/` every 5,000 steps.

### Prerequisites

`train_gpt2.py` needs two files that are not in this repository yet:

- `hellaswag.py`, which provides `render_example` and `iterate_examples` for the HellaSwag evaluation
- the tokenized FineWeb-Edu shards in `edu_fineweb10B/`, which are created by Karpathy's `fineweb.py`

Both are in [karpathy/build-nanogpt](https://github.com/karpathy/build-nanogpt).

Python dependencies: `torch`, `tiktoken`, `numpy`, `transformers` (only needed for `from_pretrained`).

### Running

```bash
cd tutorial2

# single GPU / CPU / MPS
python train_gpt2.py

# multi-GPU (e.g. 8 GPUs)
torchrun --standalone --nproc_per_node=8 train_gpt2.py
```

`torch.compile` is turned off (`use_compile = False`) because it currently breaks HellaSwag evaluation and text generation.

## References

- [karpathy/ng-video-lecture](https://github.com/karpathy/ng-video-lecture): reference code for Tutorial 1
- [karpathy/build-nanogpt](https://github.com/karpathy/build-nanogpt): reference code for Tutorial 2
- [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)
