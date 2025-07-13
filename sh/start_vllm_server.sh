#!/bin/bash

# 使用uv运行vLLM服务器
echo "Starting vLLM server with Qwen3-4B-Base model..."

# 指定使用1号GPU卡启动vLLM服务器
CUDA_VISIBLE_DEVICES=1 uv run vllm serve ./models/Qwen3-4B-Chat \
    --host 0.0.0.0 \
    --port 6688 \
    --gpu-memory-utilization 0.7 \
    --max-model-len 8192 \
    --served-model-name Qwen3-4B \
    --trust-remote-code