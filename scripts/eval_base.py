import argparse
from pathlib import Path
from typing import Literal

import torch
import yaml
from loguru import logger
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.data_loader import get_hotpot_dataloader
from utils.metrics import calculate_metrics


@torch.no_grad()
def generate_answers(
    model: AutoModelForCausalLM,
    dataloader: torch.utils.data.DataLoader,
    tokenizer: AutoTokenizer,
    device: str,
    max_new_tokens: int = 50,
):
    """
    Generates answers for the entire validation set using a standard Causal LM.
    """
    model.eval()
    predictions = []
    ground_truths = []

    for batch in tqdm(dataloader, desc="Generating Answers"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        # For generation, we only need the prompt part of the input
        prompt_end_index = (batch["labels"][0] != -100).nonzero(as_tuple=True)[0][0]
        prompt_input_ids = input_ids[:, :prompt_end_index]
        prompt_attention_mask = attention_mask[:, :prompt_end_index]

        # Get the maximum length from the model's configuration
        max_length = model.config.n_positions
        max_prompt_length = max_length - max_new_tokens

        # Truncate prompt from the left if it's too long to make space for new tokens
        if prompt_input_ids.shape[1] > max_prompt_length:
            logger.warning(
                f"Prompt length {prompt_input_ids.shape[1]} is too long, truncating to {max_prompt_length} to allow for generation."
            )
            prompt_input_ids = prompt_input_ids[:, -max_prompt_length:]
            prompt_attention_mask = prompt_attention_mask[:, -max_prompt_length:]

        # Use model.generate for simpler and more efficient generation
        generated_output = model.generate(
            input_ids=prompt_input_ids,
            attention_mask=prompt_attention_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False,  # Use greedy decoding
        )

        # The generated output contains the prompt + new tokens
        generated_answer_ids = generated_output[0][prompt_input_ids.shape[1] :]
        generated_answer = tokenizer.decode(
            generated_answer_ids, skip_special_tokens=True
        )

        ground_truth_ids = batch["input_ids"][0][prompt_input_ids.shape[1] :]
        ground_truth = tokenizer.decode(ground_truth_ids, skip_special_tokens=True)

        predictions.append(generated_answer)
        ground_truths.append(ground_truth)

    return predictions, ground_truths


def get_deivce() -> Literal["cuda", "mps", "cpu"]:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def main(config_path: str):
    # --- 1. Load Configuration ---
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    logger.info(f"Loaded configuration from {config_path}")

    # --- 2. Configuration ---
    model_config = config["model"]
    eval_config = config["evaluation"]
    paths_config = config.get("paths", {})

    MODEL_NAME = model_config["name"]
    MAX_LENGTH = model_config["max_length"]
    BATCH_SIZE = eval_config["batch_size"]
    MAX_NEW_TOKENS = eval_config["max_new_tokens"]

    if eval_config.get("device") == "auto":
        DEVICE = get_deivce()
    else:
        DEVICE = eval_config.get("device", "cpu")

    LOG_DIR = Path(paths_config.get("log_dir", "logs"))
    LOG_DIR.mkdir(exist_ok=True)

    # --- 3. Logging Setup ---
    logger.remove()  # Remove default logger
    logger.add(
        lambda msg: tqdm.write(msg, end=""), colorize=True, level="INFO"
    )  # Tqdm-friendly logger for console
    log_file_name = f"evaluation_base_{MODEL_NAME.replace('/', '_')}.log"
    logger.add(LOG_DIR / log_file_name, level="DEBUG")  # File logger
    logger.info(f"Starting evaluation of base model on device: {DEVICE}")

    # --- 4. Load Tokenizer and Model ---
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # For batch generation, decoder-only models need left-side padding
    tokenizer.padding_side = "left"

    logger.info(f"Loading base model: {MODEL_NAME}...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)

    # --- 5. Load Data ---
    logger.info("Loading validation data...")
    val_dataloader = get_hotpot_dataloader(
        "validation", tokenizer, batch_size=BATCH_SIZE, max_length=MAX_LENGTH
    )

    # --- 6. Generate and Evaluate ---
    predictions, ground_truths = generate_answers(
        model, val_dataloader, tokenizer, DEVICE, max_new_tokens=MAX_NEW_TOKENS
    )

    logger.info("Calculating metrics...")
    metrics = calculate_metrics(predictions, ground_truths)

    logger.success("Evaluation finished!")
    logger.success(f"Exact Match: {metrics['exact_match']:.2f}%")
    logger.success(f"F1 Score: {metrics['f1']:.2f}%")

    # Optionally, log some examples
    for i in range(min(5, len(predictions))):
        logger.info(f"Example {i + 1}:")
        logger.info(f"  - Ground Truth: {ground_truths[i]}")
        logger.info(f"  - Prediction:   {predictions[i]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a base LLM on HotpotQA.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default_eval_config.yaml",
        help="Path to the evaluation configuration YAML file.",
    )
    args = parser.parse_args()
    main(args.config)
