# SFT (Supervised Fine-Tuning) with LoRA for Qwen3-4B-Chat

This directory contains scripts and configurations for fine-tuning the Qwen3-4B-Chat model using LoRA (Low-Rank Adaptation) technique.

## Directory Structure

```
sft/
├── configs/
│   ├── lora_config.yaml         # LoRA training configuration
│   └── single_gpu_test.yaml     # Single GPU test configuration
├── scripts/
│   ├── train_lora.py            # Main training script
│   ├── start_training.sh        # Training startup script
│   ├── test_single_gpu.sh       # Single GPU test script
│   ├── merge_lora.py            # LoRA weight merge script
│   ├── merge_lora.sh            # Merge startup script
│   ├── quick_merge.py           # Quick merge utility
│   └── prepare_data.py          # Data preparation utility
├── data/                        # Training data directory
└── output/                      # Model output directory
```

## Quick Start

### 1. Prepare Training Data

#### Option A: Use existing training dataset
```bash
# Copy existing training data
cp ./data/training_dataset.jsonl ./sft/data/

# Or prepare data from CSV
uv run python sft/scripts/prepare_data.py \
    --input ./data/input.csv \
    --output ./sft/data/training_data.jsonl \
    --format csv \
    --instruction_col question_text
```

#### Option B: Create sample data for testing
```bash
uv run python sft/scripts/prepare_data.py \
    --output ./sft/data/sample_data.jsonl \
    --format sample \
    --num_samples 50
```

### 2. Start Training

```bash
# Start training with default configuration
bash sft/scripts/start_training.sh

# Or with custom parameters
bash sft/scripts/start_training.sh \
    ./sft/configs/lora_config.yaml \
    ./sft/data \
    ./sft/output
```

### 3. Monitor Training

Training logs will be displayed in the terminal. The model checkpoints will be saved in:
- `./sft/output/` - Full model checkpoints
- `./sft/output/lora_adapters/` - LoRA adapter weights only

## Configuration

The main configuration file is `configs/lora_config.yaml`. Key parameters:

### LoRA Settings
- `lora_rank`: 64 (rank of adaptation matrices)
- `lora_alpha`: 128 (scaling parameter)
- `lora_dropout`: 0.05 (dropout rate)
- `lora_target_modules`: Target modules for LoRA adaptation

### Training Settings
- `num_train_epochs`: 3
- `per_device_train_batch_size`: 2
- `gradient_accumulation_steps`: 8
- `learning_rate`: 5.0e-5
- `cutoff_len`: 4096 (maximum sequence length)

### Optimization
- `bf16`: true (use bfloat16 for efficiency)
- `gradient_checkpointing`: true (save memory)
- `group_by_length`: true (optimize batching)

## Data Format

Training data should be in JSONL format with the following structure:

```json
{"instruction": "What is the capital of France?", "input": "", "output": "The capital of France is Paris."}
{"instruction": "Calculate:", "input": "15 + 27", "output": "The sum of 15 and 27 is 42."}
```

Required fields:
- `instruction`: The task or question
- `input`: Additional context (can be empty)
- `output`: Expected response

## Memory Requirements

- **Minimum GPU Memory**: 16GB VRAM (for batch_size=2)
- **Recommended**: 24GB+ VRAM for larger batch sizes
- **CPU RAM**: 32GB+ recommended

## Advanced Usage

### Custom Configuration

Create a custom config file:

```yaml
# custom_config.yaml
model_name_or_path: "./models/Qwen3-4B-Chat"
lora_rank: 32  # Lower rank for less parameters
lora_alpha: 64
num_train_epochs: 5
learning_rate: 1e-4
# ... other parameters
```

Run with custom config:
```bash
bash sft/scripts/start_training.sh ./custom_config.yaml
```

### Resume Training

To resume from a checkpoint:

```bash
# Modify the config file to set:
# resume_from_checkpoint: "./sft/output/checkpoint-500"
```

### Multi-GPU Training

For multi-GPU training, use:

```bash
# Set environment variable
export CUDA_VISIBLE_DEVICES=0,1,2,3

# The training script will automatically detect multiple GPUs
bash sft/scripts/start_training.sh
```

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce `per_device_train_batch_size` or increase `gradient_accumulation_steps`
2. **Slow Training**: Enable `gradient_checkpointing` and `bf16`
3. **Data Loading Error**: Check data format and file paths

### Performance Optimization

- Use `flash_attention_2` for faster attention computation
- Enable `group_by_length` for efficient batching
- Adjust `dataloader_num_workers` based on CPU cores

## LoRA Weight Merging

After training, you can merge LoRA adapters with the base model to create a standalone fine-tuned model.

### Quick Merge (Recommended)

```bash
# Quick merge using the test LoRA adapters
uv run python sft/scripts/quick_merge.py sft/test_output/lora_adapters -o ./merged_model

# Or merge your own trained adapters
uv run python sft/scripts/quick_merge.py ./sft/output/lora_adapters -o ./my_merged_model
```

### Full Merge with Validation

```bash
# Complete merge with model validation
bash sft/scripts/merge_lora.sh \
    ./models/Qwen3-4B-Chat \
    ./sft/output/lora_adapters \
    ./sft/merged_model \
    bfloat16 \
    true

# Or use the Python script directly
uv run python sft/scripts/merge_lora.py \
    --base_model ./models/Qwen3-4B-Chat \
    --lora_adapter ./sft/output/lora_adapters \
    --output_dir ./sft/merged_model \
    --validate \
    --test_prompt "请介绍一下你的功能"
```

### Using Merged Model

Once merged, you can use the model like any standard transformers model:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load merged model
tokenizer = AutoTokenizer.from_pretrained('./sft/merged_model', trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    './sft/merged_model', 
    torch_dtype=torch.bfloat16, 
    device_map="auto",
    trust_remote_code=True
)

# Use for inference
inputs = tokenizer("你好，请介绍一下自己", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.7)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

### Deploy with vLLM

```bash
# Start vLLM server with merged model
vllm serve ./sft/merged_model --port 8000 --gpu-memory-utilization 0.8
```

## Output

After training completion:
- Model checkpoints: `./sft/output/`
- LoRA adapters: `./sft/output/lora_adapters/`
- Merged models: `./sft/merged_model/` (after merging)
- Training logs: Console output and tensorboard logs (if enabled)

The LoRA adapters can be loaded separately for inference or merged with the base model for standalone deployment.