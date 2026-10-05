import os

import lightning.pytorch as pl
import torch
import yaml
from lightning.pytorch.accelerators import TPUAccelerator
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
)
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities import rank_zero_only
from pyL_modules import CONFIG_PATH, PyLDataModule, PyLModel

# Global config
with open(CONFIG_PATH, "r") as file:
    config_training = yaml.safe_load(file)
    config_training = {k: v["value"] for k, v in config_training.items()}


def train(use_wandb=True, resume_id=None, ckpt_path=None):
    pl.seed_everything(config_training["experiment_details"]["seed"])
    torch.set_float32_matmul_precision("high")  # TF32 matmuls on Ampere+ GPUs

    # Create required directories
    model_dir = os.path.join(
        config_training["experiment_details"]["model_dir"],
        config_training["experiment_details"]["experiment_name"],
    )
    print(f"Model directory: {model_dir}")
    model_dir = os.path.join(model_dir, "checkpoints")
    os.makedirs(model_dir, exist_ok=True)

    callbacks = []
    if use_wandb:
        wandb_config = config_training["wandb_config"]
        wandb_config["notes"] = config_training["experiment_details"]["experiment_name"]
        if resume_id:
            wandb_config["id"] = resume_id
            wandb_config["resume"] = "must"
        wandb_logger = WandbLogger(**wandb_config)
        model_name = wandb_logger.experiment.name if rank_zero_only.rank == 0 else None
        wandb_logger.experiment.log_code(
            root=".",
            include_fn=lambda path: path.endswith((".py", ".yaml")),
        )
        callbacks.append(ModelSummary(max_depth=2))
        callbacks.append(LearningRateMonitor(logging_interval="step"))
    else:
        model_name = "local-test"
        wandb_logger = None

    # Create Data Module
    print("Creating Data Module")
    dataset = PyLDataModule()

    # Create Model
    print("Creating Model")
    model = PyLModel(wandb_logger=wandb_logger)

    def get_trainer_params():
        if TPUAccelerator.is_available():
            return {"accelerator": "tpu", "devices": "auto", "strategy": "auto"}
        return {"accelerator": "auto", "devices": "auto", "strategy": "auto"}

    device_params = get_trainer_params()
    print(f"Device parameters: {device_params}")

    # Gradient accumulation calculation
    hparams = config_training["training_hyperparameters"]
    B = config_training["dataset_configs"]["micro_batch_size"]
    T = config_training["dataset_configs"]["sequence_length"]
    world_size = get_world_size(device_params["accelerator"])
    assert hparams["total_batch_size"] % (B * T * world_size) == 0, (
        "make sure total_batch_size is divisible by B * T * world_size"
    )
    grad_accum_steps = hparams["total_batch_size"] // (B * T * world_size)
    print(f"Total batch size: {hparams['total_batch_size']} tokens")
    print(f"=> world size {world_size}, gradient accumulation steps {grad_accum_steps}")

    callbacks.extend(define_all_callbacks(model_dir, model_name))

    # Initialize Trainer
    trainer = pl.Trainer(
        accelerator=device_params["accelerator"],
        strategy=device_params["strategy"],
        devices=device_params["devices"],
        precision=hparams["precision"],
        max_steps=hparams["max_steps"],
        accumulate_grad_batches=grad_accum_steps,
        gradient_clip_val=1.0,
        val_check_interval=hparams["eval_interval"] * grad_accum_steps,
        check_val_every_n_epoch=None,
        limit_val_batches=hparams["val_loss_steps"],
        log_every_n_steps=1,
        callbacks=callbacks,
        logger=wandb_logger if wandb_logger is not None else False,
    )

    # Train the model
    trainer.fit(model, dataset, ckpt_path=ckpt_path)

    # Finish the experiment
    if use_wandb:
        wandb_logger.experiment.finish()


def get_world_size(accelerator):
    if accelerator == "tpu":
        return TPUAccelerator.auto_device_count()
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    return 1


def define_all_callbacks(model_dir, model_name):
    checkpoint_callback_1 = ModelCheckpoint(
        dirpath=model_dir,
        filename=f"{model_name}_best_val_loss",
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        save_last=False,
        verbose=True,
    )

    checkpoint_callback_2 = ModelCheckpoint(
        dirpath=model_dir,
        filename=f"{model_name}_latest",
        every_n_train_steps=1000,
        verbose=True,
    )

    callbacks = [
        checkpoint_callback_1,
        checkpoint_callback_2,
    ]

    return callbacks


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume_id", default=None, help="wandb run id to resume logging into"
    )
    parser.add_argument(
        "--ckpt_path", default=None, help="checkpoint path to resume training from"
    )
    args = parser.parse_args()

    train(
        use_wandb=config_training["experiment_details"]["use_wandb"],
        resume_id=args.resume_id,
        ckpt_path=args.ckpt_path,
    )
