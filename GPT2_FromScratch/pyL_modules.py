import inspect
import math
import time

import lightning.pytorch as pl
import torch
import yaml
from custom_dataset import CONFIG_PATH, ShardDataset
from lightning.pytorch.utilities import grad_norm
from model import GPT2
from torch.nn import functional as F
from torch.utils.data import DataLoader

# Global config
with open(CONFIG_PATH, "r") as file:
    _training_config = yaml.safe_load(file)
    config_training = {k: v["value"] for k, v in _training_config.items()}


class PyLDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.dataset_configs = config_training["dataset_configs"]

    def setup(self, stage=None):
        self.train_set = ShardDataset(split="train", **self.dataset_configs)
        self.val_set = ShardDataset(split="val", **self.dataset_configs)

    def _dataloader(self, dataset, shuffle, drop_last):
        return DataLoader(
            dataset,
            batch_size=self.dataset_configs["micro_batch_size"],
            shuffle=shuffle,
            drop_last=drop_last,
            pin_memory=torch.cuda.is_available(),
            num_workers=4,
            persistent_workers=True,
        )

    def train_dataloader(self):
        return self._dataloader(self.train_set, shuffle=True, drop_last=True)

    def val_dataloader(self):
        return self._dataloader(self.val_set, shuffle=False, drop_last=False)


class PyLModel(pl.LightningModule):
    def __init__(self, wandb_logger=None):
        super().__init__()
        self.save_hyperparameters(ignore=["wandb_logger"])
        self.wandb_logger = wandb_logger

        self.model = GPT2(**config_training["model_architecture_hyperparameters"])
        if config_training["training_hyperparameters"]["compile"] and torch.cuda.is_available():
            self.model.compile()
        
        self.block_size = config_training["model_architecture_hyperparameters"][
            "block_size"
        ]
        self.hparams_training = config_training["training_hyperparameters"]
        self.max_length = 32
        self.tokenizer = ShardDataset.get_tokenizer()

        self._t_last_step = None
        self._loss_accum = 0.0

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat.view(-1, y_hat.size(-1)), y.view(-1))
        # Mean loss over the micro batches of this optimizer step, logged in on_before_optimizer_step
        self._loss_accum += loss.detach() / self.trainer.accumulate_grad_batches
        return loss

    def on_before_optimizer_step(self, optimizer):
        # Loss averaged over all micro batches and devices of this optimizer step
        self.log("train/loss", self._loss_accum, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self._loss_accum = 0.0

        # Gradients here are fully accumulated but not yet clipped
        norms = grad_norm(self.model, norm_type=2)
        self.log("train/grad_norm", norms["grad_2.0_norm_total"], on_step=True)

        # Throughput: time between consecutive optimizer steps
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t_now = time.time()
        if self._t_last_step is not None:
            dt = t_now - self._t_last_step
            tokens_per_step = (
                config_training["dataset_configs"]["micro_batch_size"]
                * config_training["dataset_configs"]["sequence_length"]
                * self.trainer.accumulate_grad_batches
                * self.trainer.world_size
            )
            self.log("train/step_time_ms", dt * 1000, on_step=True)
            self.log("train/tokens_per_sec", tokens_per_step / dt, on_step=True)
        self._t_last_step = t_now

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat.view(-1, y_hat.size(-1)), y.view(-1))
        self.log(
            "val/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=x.size(0),
        )
        return loss

    def on_validation_epoch_end(self):
        # Skip the sanity check and step 0, where samples are pure noise
        if self.trainer.sanity_checking or self.global_step == 0:
            return
        samples = self.generate_samples()
        if self.trainer.is_global_zero:
            for i, sample in enumerate(samples):
                print(f"step {self.global_step} sample {i}: {sample}")
            if self.wandb_logger is not None:
                self.wandb_logger.log_text(
                    key="samples",
                    columns=["step", "sample"],
                    data=[[self.global_step, s] for s in samples],
                    step=self.global_step,
                )

    def on_validation_end(self):
        # Don't count validation time towards the next step's throughput
        self._t_last_step = None

    @torch.no_grad()
    def generate_samples(self):
        tokens = torch.tensor(self.tokenizer.encode("ROMEO:"), dtype=torch.long)
        xgen = tokens.unsqueeze(0).repeat(3, 1)
        xgen = xgen.to(self.device)
        sample_rng = torch.Generator(device=self.device)
        sample_rng.manual_seed(42 + self.global_rank)

        self.model.eval()
        while xgen.size(1) < self.max_length:
            # crop to the last block_size tokens
            # call forward directly to skip the compiled path, which would recompile for every new length
            logits = self.model.forward(xgen[:, -self.block_size :])  # (B, T, vocab_size)
            # take the logits at the last position
            logits = logits[:, -1, :]  # (B, vocab_size)
            probs = F.softmax(logits, dim=-1)
            # top-k sampling (huggingface pipeline default is k=50)
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            # multinomial does not require the input to sum to 1
            ix = torch.multinomial(topk_probs, 1, generator=sample_rng)  # (B, 1)
            xcol = torch.gather(topk_indices, -1, ix)  # (B, 1)
            xgen = torch.cat((xgen, xcol), dim=1)
        self.model.train()

        return [self.tokenizer.decode(row.tolist()) for row in xgen]

    def get_lr_multiplier(self, step):
        warmup_steps = self.hparams_training["warmup_steps"]
        max_steps = self.hparams_training["max_steps"]
        min_ratio = self.hparams_training["min_lr"] / self.hparams_training["max_lr"]
        # 1) linear warmup
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        # 2) past max_steps, hold at min_lr
        if step > max_steps:
            return min_ratio
        # 3) in between, cosine decay down to min_lr
        decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) 
        return min_ratio + coeff * (1.0 - min_ratio)

    def configure_optimizers(self):
        # Weight decay only on 2D params (matmul weights + embeddings), not on biases and layernorm params
        params = [p for p in self.model.parameters() if p.requires_grad]
        decay_params = [p for p in params if p.dim() >= 2]
        nodecay_params = [p for p in params if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": self.hparams_training["weight_decay"]},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        # Use the fused AdamW kernel when available on CUDA
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and self.device.type == "cuda"

        if self.trainer.is_global_zero:
            print(
                f"num decayed parameter tensors: {len(decay_params)}, "
                f"with {sum(p.numel() for p in decay_params):,} parameters"
            )
            print(
                f"num non-decayed parameter tensors: {len(nodecay_params)}, "
                f"with {sum(p.numel() for p in nodecay_params):,} parameters"
            )
            print(f"using fused AdamW: {use_fused}")

        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=self.hparams_training["max_lr"],
            betas=tuple(self.hparams_training["adam_betas"]),
            eps=self.hparams_training["adam_eps"],
            fused=use_fused,
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=self.get_lr_multiplier
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


if __name__ == "__main__":
    dataset = PyLDataModule()
    model = PyLModel()
    breakpoint()
