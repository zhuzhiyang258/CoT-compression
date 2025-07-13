#!/bin/bash

# SFT Training Startup Script for Qwen3-4B-Chat with LoRA
# Usage: bash sft/scripts/start_training.sh [CONFIG_PATH] [DATA_DIR] [OUTPUT_DIR]

set -e  # Exit on any error

# Default paths
DEFAULT_CONFIG="./sft/configs/lora_config.yaml"
DEFAULT_DATA_DIR="./sft/data"
DEFAULT_OUTPUT_DIR="./sft/output"

# Parse command line arguments
CONFIG_PATH=${1:-$DEFAULT_CONFIG}
DATA_DIR=${2:-$DEFAULT_DATA_DIR}
OUTPUT_DIR=${3:-$DEFAULT_OUTPUT_DIR}

echo "==================== SFT Training Setup ===================="
echo "Config file: $CONFIG_PATH"
echo "Data directory: $DATA_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "=============================================================="

# Check if UV is available
if ! command -v uv &> /dev/null; then
    echo "Error: UV package manager not found. Please install UV first."
    echo "Run: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Check if config file exists
if [ ! -f "$CONFIG_PATH" ]; then
    echo "Error: Config file not found: $CONFIG_PATH"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Check if training data exists
if [ ! -d "$DATA_DIR" ]; then
    echo "Warning: Data directory not found: $DATA_DIR"
    echo "Creating data directory..."
    mkdir -p "$DATA_DIR"
    
    # Check if training dataset exists in the main data directory
    if [ -f "./data/training_dataset.jsonl" ]; then
        echo "Found training dataset in ./data/training_dataset.jsonl"
        echo "You may want to copy or link it to the SFT data directory:"
        echo "  cp ./data/training_dataset.jsonl $DATA_DIR/"
    else
        echo "Error: No training data found."
        echo "Please prepare your training data in JSONL format with the following structure:"
        echo '{"instruction": "question", "input": "optional_input", "output": "expected_answer"}'
        echo "And place it in: $DATA_DIR/"
        exit 1
    fi
fi

# Check GPU availability
echo "Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits
    echo ""
else
    echo "Warning: nvidia-smi not found. GPU status unknown."
fi

# Set environment variables for training
export CUDA_VISIBLE_DEVICES=0  # Use GPU 0 for training
export TOKENIZERS_PARALLELISM=false  # Avoid tokenizer warnings
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

echo "Starting SFT training..."
echo "Command: uv run python sft/scripts/train_lora.py --config $CONFIG_PATH --data_dir $DATA_DIR --output_dir $OUTPUT_DIR"
echo ""

# Start training with UV
uv run python sft/scripts/train_lora.py \
    --config "$CONFIG_PATH" \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "==================== Training Completed ===================="
echo "Model saved to: $OUTPUT_DIR"
echo "LoRA adapters saved to: $OUTPUT_DIR/lora_adapters"
echo "=============================================================="