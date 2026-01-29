# GRPO Countdown Problem

A project for training language models to solve arithmetic countdown problems using Supervised Fine-Tuning (SFT) followed by Group Relative Policy Optimization (GRPO).

## Overview

This project implements a two-stage training pipeline:

1. **SFT (Supervised Fine-Tuning)**: Train the model on arithmetic problems with correct solutions
2. **GRPO (Group Relative Policy Optimization)**: Further optimize the model using reward-based learning

The goal is to train a language model to solve arithmetic countdown problems where you must use exactly four given numbers with basic arithmetic operations (+, -, *, /) to reach a target value.

## Project Structure

```
grpo-countdown-problem/
├── data/                           # Training and test datasets
├── models/                         # Saved model checkpoints
│   ├── sft/                       # SFT model outputs
│   └── grpo/                      # GRPO model outputs
├── src/
│   ├── config/                    # Configuration files
│   │   ├── grpo/                  # GRPO training configs
│   │   └── sft/                   # SFT training configs
│   ├── dataset/                   # Dataset loading and processing
│   ├── examples/                  # Example scripts for inference
│   ├── scripts/                   # Data generation and processing
│   ├── training/                  # Training scripts
│   │   ├── grpo/                  # GRPO training
│   │   └── sft/                   # SFT training
│   └── utils/                     # Utility functions
├── main.py                        # Main entry point
├── pyproject.toml                 # Project dependencies
└── README.md                      # This file
```

## Requirements

- Python 3.12+
- CUDA-capable GPU (recommended)
- At least 8GB GPU memory for Qwen2.5-Math-1.5B model

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd grpo-countdown-problem
   ```

2. **Install dependencies using uv (recommended):**
   ```bash
   # Install uv if you haven't already
   curl -LsSf https://astral.sh/uv/install.sh | sh
   
   # Install project dependencies
   uv sync
   ```

   **Or using pip:**
   ```bash
   pip install -e .
   ```

3. **Set up environment variables (if using OpenAI for data generation):**
   ```bash
   cp .env.example .env
   # Edit .env and add your OpenAI API key
   ```

## Data Preparation

### Generate Training Data

1. **Generate SFT training data:**
   ```bash
   python src/scripts/generate_training_dataset_sft.py \
     --output_path data/sft/train.csv \
     --num_problems 10000 \
     --num_workers 4
   ```

2. **Generate GRPO training data:**
   ```bash
   python src/scripts/generate_training_dataset_grpo.py \
     --output_path data/grpo/train.csv \
     --num_problems 10000 \
     --num_workers 4
   ```

3. **Generate test data:**
   ```bash
   python src/scripts/generate_training_dataset_grpo.py \
     --output_path data/grpo/test.csv \
     --num_problems 1000 \
     --num_workers 4
   ```

### Data Format

The CSV files contain the following columns:
- `id`: Unique problem identifier
- `problem_description`: Natural language description of the problem
- `correct_answer`: The target arithmetic expression
- `num1`, `num2`, `num3`, `num4`: The four numbers to use
- `reasoning` (SFT only): Step-by-step solution explanation

## Training

### Stage 1: Supervised Fine-Tuning (SFT)

Train the base model on arithmetic problems with supervised learning:

```bash
python src/training/sft/train_sft_hydra.py
```

**Configuration:** The training uses Hydra configuration files in `src/config/sft/`:
- `config.yaml`: Main configuration
- `dataset/default.yaml`: Dataset settings
- `model/qwen2.5-3b.yaml`: Model and LoRA settings
- `training/default.yaml`: Training hyperparameters

**Key parameters:**
- Base model: `Qwen/Qwen2.5-Math-1.5B`
- LoRA rank: 64
- Learning rate: 2e-5
- Batch size: 4 (per device)
- Epochs: 2

**Output:** Trained SFT model saved to `models/sft/`

### Stage 2: Group Relative Policy Optimization (GRPO)

Further optimize the SFT model using reward-based learning:

```bash
python src/training/grpo/train_grpo_hydra.py
```

**Configuration:** Uses Hydra configuration files in `src/config/grpo/`:
- `config.yaml`: Main configuration (includes SFT model path)
- `dataset/default.yaml`: Dataset settings
- `model/qwen2.5-3b.yaml`: Model and LoRA settings  
- `training/default.yaml`: Training hyperparameters

**Key parameters:**
- Builds on SFT model from `models/sft/`
- Learning rate: 1e-5
- Batch size: 2 (per device)
- Epochs: 1
- Generations per prompt: 8
- Reward function: Mathematical correctness

**Output:** Trained GRPO model saved to `models/grpo/`

### Custom Configuration

You can override configuration parameters:

```bash
# Override dataset size
python src/training/sft/train_sft_hydra.py dataset.max_rows=5000

# Override learning rate and batch size
python src/training/grpo/train_grpo_hydra.py \
  training.learning_rate=5e-6 \
  training.per_device_train_batch_size=1

# Use different output directory
python src/training/sft/train_sft_hydra.py output_dir=models/sft_experiment
```

## Inference

### Interactive Problem Solving

Use the trained model to solve individual problems:

```bash
python src/examples/run_model.py
```

This will load both SFT and GRPO models and solve a sample problem.

### Batch Evaluation

Evaluate model accuracy on a test dataset:

```bash
python src/examples/calculate_accuracy.py \
  --csv_path data/grpo/test.csv \
  --sft_model_path models/sft/ \
  --grpo_model_path models/grpo/ \
  --max_samples 100 \
  --output_path results/evaluation_results.csv
```

**Parameters:**
- `--csv_path`: Path to test CSV file
- `--sft_model_path`: Path to SFT model directory
- `--grpo_model_path`: Path to GRPO model directory
- `--max_samples`: Limit number of test samples (optional)
- `--output_path`: Save detailed results to CSV (optional)
- `--temperature`: Sampling temperature (default: 1.0)
- `--max_new_tokens`: Maximum tokens to generate (default: 4096)

**Evaluation Metrics:**
- **Accuracy**: Percentage of problems solved correctly
- **Valid Format Rate**: Percentage of responses in valid arithmetic format
- **Uses All Numbers Rate**: Percentage of responses using all four numbers

### Model-only Evaluation

Evaluate specific model stages:

```bash
# Evaluate only SFT model (no GRPO)
python src/examples/calculate_accuracy.py \
  --csv_path data/grpo/test.csv \
  --sft_model_path models/sft/ \
  --no_grpo

# Evaluate only base model (no SFT or GRPO)
python src/examples/calculate_accuracy.py \
  --csv_path data/grpo/test.csv \
  --no_sft --no_grpo
```

## Configuration Details

### Model Configuration

The project uses **Qwen2.5-Math-1.5B** as the base model with LoRA (Low-Rank Adaptation) for efficient fine-tuning:

- **LoRA rank**: 64
- **LoRA alpha**: 128
- **Target modules**: All attention and MLP layers
- **LoRA dropout**: 0.1

### Training Configuration

**SFT Training:**
- **Optimizer**: AdamW 8-bit
- **Learning rate**: 2e-5 with linear scheduler
- **Warmup ratio**: 0.1
- **Weight decay**: 0.01
- **Max sequence length**: 4096

**GRPO Training:**
- **Optimizer**: AdamW 8-bit
- **Learning rate**: 1e-5 with cosine scheduler
- **Warmup ratio**: 0.1
- **Weight decay**: 0.0
- **Temperature**: 1.0
- **Generations per prompt**: 8

## Monitoring Training

Both training scripts log to TensorBoard:

```bash
# View training logs
tensorboard --logdir models/sft/runs    # For SFT training
tensorboard --logdir models/grpo/runs   # For GRPO training
```

## Example Problem

**Input:** "Use 53, 3, 47, and 36 exactly once each with only +, -, *, and / operators to create an expression equal to 133."

**Expected Output:** A valid arithmetic expression like `53 + 47 + 36 - 3`
