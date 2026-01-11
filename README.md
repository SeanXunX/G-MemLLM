# G-MemLLM: Gated Latent Memory Augmentation for Long-Context Reasoning in Large Language Models

This project introduces **G-MemLLM**, a memory-augmented architecture that integrates a frozen LLM backbone with a trainable **Latent Memory Bank**. It is designed to enhance the multi-hop reasoning and relational knowledge retention of Large Language Models.

The key innovation is a **GRU-style gated update logic** that allows the model to selectively update, preserve, or overwrite latent memory slots, preventing the vanishing gradients of knowledge common in recurrent systems.

This repository contains the implementation for the paper: *G-G-MemLLM: Gated Latent Memory Augmentation for Long-Context Reasoning in Large Language Models*.

## Features

*   **Gated Latent Memory:** A `G-MemLLM` model that integrates a trainable memory bank with a frozen pretrained LLM.
*   **GRU-Style Gating:** A gated update mechanism to prevent memory drift and information overwrite.
*   **Composite Training Objective:** A combination of a primary task loss with sparsity and entropy regularization to ensure efficient and diverse memory slot utilization.
*   **Scalable Evaluation:** Evaluated on models from GPT-2 (124M) to Llama 3.1 (8B).
*   **Multi-Dataset Benchmarking:** Scripts to train and evaluate on **HotpotQA** (multi-hop reasoning) and **ZsRE** (zero-shot relation extraction).
*   **Configuration-driven:** Easily configurable training and evaluation parameters using YAML files.

## Key Results

G-MemLLM significantly enhances multi-hop reasoning and relational precision across model scales.
- **HotpotQA:** Achieves a **+8.56 F1** increase for GPT-2 and a **+6.89 Supporting Fact F1** increase for Llama 3.1-8B.
- **ZsRE:** Achieves a **13.3% accuracy boost** for Llama 3.1-8B.

## Installation

This project strongly recommends using `uv` as the Python package manager due to its speed and efficiency.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/SeanXunX/G-MemLLM.git
    ```

2.  **Install dependencies using `uv`:**
    ```bash
    uv sync
    ```

## Usage

To train and evaluate the model, use instructions following the format:

```bash
uv run -m <dir>.<filename> [--config <yaml file>]
```

e.g.
```bash
uv run -m scripts.eval_base --config configs/default_config.yaml
```

## Contributing

**Pre-commit Checks:** This project uses pre-commit hooks to maintain code quality and consistency. These checks are also enforced in the GitHub Actions workflow (`.github/workflows/pre-commit.yml`). Please ensure your changes pass all pre-commit checks before submitting a pull request. You can simply run the command directly to format all files:

```bash
uvx pre-commit run --all-files
```
