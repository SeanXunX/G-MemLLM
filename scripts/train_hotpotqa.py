import argparse
import os
from pathlib import Path
from typing import Literal

import torch
import yaml
from dotenv import load_dotenv
from loguru import logger
from tqdm import tqdm
from transformers import AutoTokenizer

from models.mem_llm import MemLLM
from utils.data_loader import get_hotpot_dataloader
from utils.trainer import MemoryTrainer


def get_deivce() -> Literal["cuda", "mps", "cpu"]:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


class EarlyStopper:
    """
    A simple class to handle early stopping.
    """

    def __init__(self, patience: int = 5, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")

    def __call__(self, val_loss: float) -> bool:
        """
        Returns True if training should stop, False otherwise.
        """
        if self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            logger.info("Validation loss improved.")
            return False
        else:
            self.counter += 1
            logger.info(
                f"Validation loss did not improve. Counter: {self.counter}/{self.patience}"
            )
            if self.counter >= self.patience:
                return True
        return False


@torch.no_grad()
def evaluate(
    model: MemLLM,
    dataloader: torch.utils.data.DataLoader,
    trainer: MemoryTrainer,
    device: str,
) -> float:
    """
    Runs a validation loop and returns the average loss.
    """
    model.eval()
    total_loss = 0

    for batch in tqdm(dataloader, desc="Evaluating"):
        # Reset memory for each batch to ensure independent evaluation
        model.current_memory = None

        batch_on_device = {k: v.to(device) for k, v in batch.items()}
        outputs = model(
            input_ids=batch_on_device["input_ids"],
            attention_mask=batch_on_device["attention_mask"],
        )
        loss = trainer.compute_loss(outputs, batch_on_device["labels"])
        total_loss += loss.item()

    return total_loss / len(dataloader)


def main(config_path: str):
    # --- 1. Load Configuration ---
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    logger.info(f"Loaded configuration from {config_path}")

    # --- 2. Load Environment Variables ---
    load_dotenv()
    logger.info("Loaded environment variables from .env file.")

    # --- 3. Environment and Path Configuration ---
    paths_config = config.get("paths", {})
    HF_HOME = os.getenv("HF_HOME", paths_config.get("hf_home", "./hf_cache"))
    CHECKPOINT_DIR = Path(
        os.getenv("CHECKPOINT_DIR", paths_config.get("checkpoint_dir", "checkpoints"))
    )
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    LOG_DIR = Path(os.getenv("LOG_DIR", paths_config.get("log_dir", "logs")))
    LOG_DIR.mkdir(exist_ok=True)

    # --- 4. Logging Setup (Loguru) ---
    logger.remove()  # Remove default logger
    logger.add(
        lambda msg: tqdm.write(msg, end=""), colorize=True, level="INFO"
    )  # Tqdm-friendly logger
    logger.add(LOG_DIR / "training_{time}.log", level="DEBUG")  # File logger
    logger.info("Starting training script...")

    # --- 5. Training Configuration ---
    model_config = config["model"]
    train_config = config["training"]
    early_stop_config = config["early_stopping"]

    MODEL_NAME = model_config["name"]
    MAX_LENGTH = model_config["max_length"]
    BATCH_SIZE = train_config["batch_size"]
    LEARNING_RATE = train_config["learning_rate"]
    NUM_EPOCHS = train_config["num_epochs"]

    if train_config.get("device") == "auto":
        DEVICE = get_deivce()
    else:
        DEVICE = train_config.get("device", "cpu")

    logger.info(f"Using device: {DEVICE}")
    logger.info(f"Hugging Face cache directory: {HF_HOME}")

    # --- Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=HF_HOME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- DataLoaders ---
    logger.info("Loading datasets...")
    train_dataloader = get_hotpot_dataloader(
        "train", tokenizer, batch_size=BATCH_SIZE, max_length=MAX_LENGTH
    )
    val_dataloader = get_hotpot_dataloader(
        "validation", tokenizer, batch_size=BATCH_SIZE, max_length=MAX_LENGTH
    )

    # --- Model, Trainer, and Early Stopper ---
    logger.info("Initializing model, trainer, and early stopper...")
    model = MemLLM(MODEL_NAME).to(DEVICE)
    trainer = MemoryTrainer(model, learning_rate=LEARNING_RATE)
    early_stopper = EarlyStopper(
        patience=early_stop_config["patience"], min_delta=early_stop_config["min_delta"]
    )

    # --- 6. Load from Checkpoint (if available) ---
    start_epoch = 0
    latest_checkpoint = CHECKPOINT_DIR / "latest_checkpoint.pt"
    if latest_checkpoint.exists():
        logger.info(f"Loading checkpoint from {latest_checkpoint}...")
        checkpoint = torch.load(latest_checkpoint, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        early_stopper.best_loss = checkpoint["best_val_loss"]
        logger.info(
            f"Resuming training from epoch {start_epoch}. Best validation loss: {early_stopper.best_loss:.4f}"
        )

    # --- 7. Training Loop ---
    for epoch in range(start_epoch, NUM_EPOCHS):
        logger.info(f"--- Starting Epoch {epoch + 1}/{NUM_EPOCHS} ---")
        model.train()
        model.current_memory = None

        train_pbar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1} Training")
        for batch in train_pbar:
            batch_on_device = {k: v.to(DEVICE) for k, v in batch.items()}
            loss = trainer.train_step(batch_on_device)
            train_pbar.set_postfix({"Loss": f"{loss:.4f}"})

        # --- Validation ---
        val_loss = evaluate(model, val_dataloader, trainer, DEVICE)
        logger.info(f"Epoch {epoch + 1} finished. Validation Loss: {val_loss:.4f}")

        # --- 8. Checkpointing ---
        if val_loss < early_stopper.best_loss:
            logger.success(
                f"New best validation loss: {val_loss:.4f}. Saving checkpoint."
            )
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": trainer.optimizer.state_dict(),
                "best_val_loss": val_loss,
            }
            torch.save(checkpoint, CHECKPOINT_DIR / f"epoch_{epoch + 1}.pt")
            torch.save(
                checkpoint, latest_checkpoint
            )  # Overwrite latest for easy resume

        # --- 9. Early Stopping ---
        if early_stopper(val_loss):
            logger.warning("Early stopping triggered. Halting training.")
            break

    logger.success("Training finished.")

    # --- Save Final Model Weights ---
    final_model_path = "hotpotqa_memory_module_final.pt"
    trainable_state_dict = {
        k: v for k, v in model.state_dict().items() if v.requires_grad
    }
    torch.save(trainable_state_dict, final_model_path)
    logger.success(f"Saved final trainable memory module weights to {final_model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a memory-augmented LLM on HotpotQA."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default_config.yaml",
        help="Path to the training configuration YAML file.",
    )
    args = parser.parse_args()
    main(args.config)
