#!/bin/bash

# LoRA权重合并启动脚本
# 用法: bash sft/scripts/merge_lora.sh [LORA_ADAPTER_PATH] [OUTPUT_DIR] [OPTIONS]

set -e

echo "🔗 LoRA权重合并工具"
echo "============================================"

# 默认参数
DEFAULT_BASE_MODEL="./models/Qwen3-4B-Chat"
DEFAULT_LORA_ADAPTER="./sft/outputs/training/test_run/lora_adapters"
DEFAULT_OUTPUT_DIR="./sft/outputs/merged_models/default"
DEFAULT_TORCH_DTYPE="bfloat16"

# 解析命令行参数
BASE_MODEL=${1:-$DEFAULT_BASE_MODEL}
LORA_ADAPTER=${2:-$DEFAULT_LORA_ADAPTER}
OUTPUT_DIR=${3:-$DEFAULT_OUTPUT_DIR}
TORCH_DTYPE=${4:-$DEFAULT_TORCH_DTYPE}
VALIDATE=${5:-"true"}

echo "配置参数:"
echo "  基础模型: $BASE_MODEL"
echo "  LoRA适配器: $LORA_ADAPTER"
echo "  输出目录: $OUTPUT_DIR"
echo "  数据类型: $TORCH_DTYPE"
echo "  验证模型: $VALIDATE"
echo "============================================"

# 检查依赖
if ! command -v uv &> /dev/null; then
    echo "❌ UV包管理器未找到，请先安装UV"
    echo "安装命令: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# 检查基础模型
if [ ! -d "$BASE_MODEL" ]; then
    echo "❌ 基础模型目录不存在: $BASE_MODEL"
    exit 1
fi

# 检查LoRA适配器
if [ ! -d "$LORA_ADAPTER" ]; then
    echo "❌ LoRA适配器目录不存在: $LORA_ADAPTER"
    echo "请先完成LoRA训练，或指定正确的适配器路径"
    exit 1
fi

# 检查适配器配置文件
if [ ! -f "$LORA_ADAPTER/adapter_config.json" ]; then
    echo "❌ LoRA适配器配置文件缺失: $LORA_ADAPTER/adapter_config.json"
    exit 1
fi

# 检查GPU状态
echo "🎮 GPU状态检查..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits | head -2
    echo ""
else
    echo "⚠️  nvidia-smi不可用，无法检查GPU状态"
fi

# 设置环境变量
export CUDA_VISIBLE_DEVICES=1  # 使用GPU 1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

echo "🚀 开始LoRA权重合并..."

# 构建合并命令
MERGE_CMD="uv run python sft/scripts/merge_lora.py"
MERGE_CMD="$MERGE_CMD --base_model \"$BASE_MODEL\""
MERGE_CMD="$MERGE_CMD --lora_adapter \"$LORA_ADAPTER\""
MERGE_CMD="$MERGE_CMD --output_dir \"$OUTPUT_DIR\""
MERGE_CMD="$MERGE_CMD --torch_dtype \"$TORCH_DTYPE\""

if [ "$VALIDATE" = "true" ]; then
    MERGE_CMD="$MERGE_CMD --validate"
    MERGE_CMD="$MERGE_CMD --test_prompt \"请介绍一下自己，并解释你的功能。\""
fi

echo "执行命令: $MERGE_CMD"
echo ""

# 执行合并
eval $MERGE_CMD

# 检查合并结果
if [ $? -eq 0 ]; then
    echo ""
    echo "============================================"
    echo "✅ LoRA权重合并完成！"
    echo ""
    echo "📁 合并后模型位置: $OUTPUT_DIR"
    echo ""
    echo "📋 输出目录内容:"
    ls -la "$OUTPUT_DIR" | head -10
    echo ""
    
    # 计算模型大小
    if command -v du &> /dev/null; then
        MODEL_SIZE=$(du -sh "$OUTPUT_DIR" | cut -f1)
        echo "📦 模型大小: $MODEL_SIZE"
    fi
    
    echo ""
    echo "🎯 使用合并后的模型:"
    echo "1. 推理测试:"
    echo "   python -c \"from transformers import AutoTokenizer, AutoModelForCausalLM; tokenizer = AutoTokenizer.from_pretrained('$OUTPUT_DIR', trust_remote_code=True); model = AutoModelForCausalLM.from_pretrained('$OUTPUT_DIR', trust_remote_code=True)\""
    echo ""
    echo "2. vLLM服务器:"
    echo "   vllm serve '$OUTPUT_DIR' --port 8000"
    echo ""
    echo "============================================"
else
    echo ""
    echo "❌ LoRA权重合并失败！"
    echo "请检查错误信息并重试"
    exit 1
fi