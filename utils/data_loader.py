from typing import Literal

from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset


def format_hotpot_example(example, is_training=True):
    """Formats a HotpotQA example into a single string for the LLM."""
    # Concatenate all paragraphs from the context into a single string
    context_str = ""
    for title, sentences in zip(
        example["context"]["title"], example["context"]["sentences"]
    ):
        # Filter out empty titles or sentence lists
        if title and sentences:
            context_str += f"Title: {title}\n{' '.join(sentences)}\n"

    # Define the structure for the LLM input
    # The model learns to generate the answer after seeing the [ANSWER] token.
    formatted_str = (
        f"[CONTEXT]\n{context_str.strip()}\n\n"
        f"[QUESTION]\n{example['question']}\n\n"
        f"[ANSWER]\n"
    )

    # For training, append the ground-truth answer to be learned
    if is_training:
        formatted_str += f"{example['answer']}"

    return formatted_str


class HotpotQADataset(Dataset):
    """
    PyTorch Dataset for the HotpotQA dataset, structured for a generative QA task.
    """

    def __init__(
        self, split: Literal["train", "validation"], tokenizer, max_length: int = 1024
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length

        print(f"Loading HotpotQA '{split}' split...")

        self.dataset = load_dataset("hotpot_qa", "distractor", split=split)

        # Using a subset for demonstration purposes to speed things up.
        # Remove `.select(range(1000))` to use the full dataset.
        # self.dataset = load_dataset("hotpot_qa", "distractor", split=split).select(range(1000))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]

        # Create the full string with the answer for the input
        full_formatted_str = format_hotpot_example(example, is_training=True)

        # Create the prompt string (without answer) to identify where labels should start
        prompt_str = format_hotpot_example(example, is_training=False)

        # Tokenize the full string to get input_ids and attention_mask
        full_tokenized = self.tokenizer(
            full_formatted_str,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = full_tokenized.input_ids.squeeze(0)
        attention_mask = full_tokenized.attention_mask.squeeze(0)

        # Tokenize the prompt just to find its length in tokens
        prompt_token_len = (
            self.tokenizer(
                prompt_str,
                max_length=self.max_length,
                truncation=True,
                return_tensors="pt",
            )
            .input_ids.squeeze(0)
            .ne(self.tokenizer.pad_token_id)
            .sum()
            .item()
        )

        # Create labels: a copy of input_ids where the prompt and padding are masked
        labels = input_ids.clone()
        # Mask the prompt part by setting its token labels to -100
        labels[: prompt_token_len - 1] = -100
        # Mask padding tokens
        labels[input_ids == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def get_hotpot_dataloader(split, tokenizer, batch_size=4, max_length=1024):
    """Creates a DataLoader for the HotpotQA dataset."""
    dataset = HotpotQADataset(split, tokenizer, max_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=(split == "train"))
