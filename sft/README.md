# SFT (Supervised Fine-Tuning) with LoRA for Qwen3-4B-Chat

本目录包含使用LoRA（低秩适应）技术对Qwen3-4B-Chat模型进行微调的脚本和配置文件。

## 目录结构

```
sft/
├── configs/
│   ├── lora_config.yaml         # LoRA训练配置
│   ├── single_gpu_test.yaml     # 单GPU测试配置
│   └── financial_training.yaml  # 金融领域训练配置
├── scripts/
│   ├── train_lora.py            # 主训练脚本
│   ├── start_training.sh        # 训练启动脚本
│   ├── test_single_gpu.sh       # 单GPU测试脚本
│   ├── merge_lora.py            # LoRA权重合并脚本
│   ├── merge_lora.sh            # 合并启动脚本
│   ├── quick_merge.py           # 快速合并工具
│   └── prepare_data.py          # 数据准备工具
├── data/                        # 训练数据目录
└── outputs/                     # 统一输出目录
    ├── training/                # 训练输出
    │   ├── default/             # 默认训练运行
    │   ├── test_run/            # 测试训练运行
    │   └── financial/           # 金融领域训练
    └── merged_models/           # 合并模型输出
        └── default/             # 默认合并模型
```

## 快速开始

### 1. 准备训练数据

#### 选项A：使用现有训练数据集
```bash
# 复制现有训练数据
cp ./data/training_dataset.jsonl ./sft/data/

# 或从CSV准备数据
uv run python sft/scripts/prepare_data.py \
    --input ./data/input.csv \
    --output ./sft/data/training_data.jsonl \
    --format csv \
    --instruction_col question_text
```

#### 选项B：创建测试用样本数据
```bash
uv run python sft/scripts/prepare_data.py \
    --output ./sft/data/sample_data.jsonl \
    --format sample \
    --num_samples 50
```

### 2. 开始训练

```bash
# 使用默认配置开始训练
bash sft/scripts/start_training.sh

# 或使用自定义参数
bash sft/scripts/start_training.sh \
    ./sft/configs/lora_config.yaml \
    ./sft/data \
    ./sft/output
```

### 3. 监控训练

训练日志将在终端中显示。模型检查点将保存在：
- `./sft/output/` - 完整模型检查点
- `./sft/output/lora_adapters/` - 仅LoRA适配器权重

## 配置

主配置文件是`configs/lora_config.yaml`。关键参数：

### LoRA设置
- `lora_rank`: 64（适应矩阵的秩）
- `lora_alpha`: 128（缩放参数）
- `lora_dropout`: 0.05（丢弃率）
- `lora_target_modules`: LoRA适应的目标模块

### 训练设置
- `num_train_epochs`: 3
- `per_device_train_batch_size`: 2
- `gradient_accumulation_steps`: 8
- `learning_rate`: 5.0e-5
- `cutoff_len`: 4096（最大序列长度）

### 优化
- `bf16`: true（使用bfloat16提高效率）
- `gradient_checkpointing`: true（节省内存）
- `group_by_length`: true（优化批处理）

## 数据格式

训练数据应为JSONL格式，具有以下结构：

```json
{"instruction": "法国的首都是什么？", "input": "", "output": "法国的首都是巴黎。"}
{"instruction": "计算：", "input": "15 + 27", "output": "15和27的和是42。"}
```

必需字段：
- `instruction`: 任务或问题
- `input`: 额外上下文（可以为空）
- `output`: 期望的响应

## 内存需求

- **最小GPU内存**: 16GB VRAM（batch_size=2）
- **推荐**: 24GB+VRAM用于更大的batch size
- **CPU内存**: 推荐32GB+

## 高级用法

### 自定义配置

创建自定义配置文件：

```yaml
# custom_config.yaml
model_name_or_path: "./models/Qwen3-4B-Chat"
lora_rank: 32  # 更低的秩以减少参数
lora_alpha: 64
num_train_epochs: 5
learning_rate: 1e-4
# ... 其他参数
```

使用自定义配置运行：
```bash
bash sft/scripts/start_training.sh ./custom_config.yaml
```

### 恢复训练

从检查点恢复：

```bash
# 修改配置文件设置：
# resume_from_checkpoint: "./sft/output/checkpoint-500"
```

### 多GPU训练

多GPU训练使用：

```bash
# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 训练脚本会自动检测多GPU
bash sft/scripts/start_training.sh
```

## 故障排除

### 常见问题

1. **内存不足**: 减少`per_device_train_batch_size`或增加`gradient_accumulation_steps`
2. **训练缓慢**: 启用`gradient_checkpointing`和`bf16`
3. **数据加载错误**: 检查数据格式和文件路径

### 性能优化

- 使用`flash_attention_2`加速注意力计算
- 启用`group_by_length`进行高效批处理
- 根据CPU核心数调整`dataloader_num_workers`

## LoRA权重合并

训练完成后，您可以将LoRA适配器与基础模型合并，创建独立的微调模型。

### 快速合并（推荐）

```bash
# 使用测试LoRA适配器进行快速合并
uv run python sft/scripts/quick_merge.py sft/test_output/lora_adapters -o ./merged_model

# 或合并您自己训练的适配器
uv run python sft/scripts/quick_merge.py ./sft/output/lora_adapters -o ./my_merged_model
```

### 完整合并与验证

```bash
# 带模型验证的完整合并
bash sft/scripts/merge_lora.sh \
    ./models/Qwen3-4B-Chat \
    ./sft/output/lora_adapters \
    ./sft/merged_model \
    bfloat16 \
    true

# 或直接使用Python脚本
uv run python sft/scripts/merge_lora.py \
    --base_model ./models/Qwen3-4B-Chat \
    --lora_adapter ./sft/output/lora_adapters \
    --output_dir ./sft/merged_model \
    --validate \
    --test_prompt "请介绍一下你的功能"
```

### 使用合并模型

合并后，您可以像使用任何标准transformers模型一样使用该模型：

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 加载合并模型
tokenizer = AutoTokenizer.from_pretrained('./sft/merged_model', trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    './sft/merged_model', 
    torch_dtype=torch.bfloat16, 
    device_map="auto",
    trust_remote_code=True
)

# 用于推理
inputs = tokenizer("你好，请介绍一下自己", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.7)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

### 使用vLLM部署

```bash
# 使用合并模型启动vLLM服务器
vllm serve ./sft/merged_model --port 8000 --gpu-memory-utilization 0.8
```

## 输出

训练完成后：
- 训练输出：`./sft/outputs/training/{run_name}/`
- LoRA适配器：`./sft/outputs/training/{run_name}/lora_adapters/`
- 合并模型：`./sft/outputs/merged_models/{model_name}/`
- 训练日志：控制台输出和tensorboard日志（如果启用）

LoRA适配器可以单独加载用于推理，或与基础模型合并用于独立部署。