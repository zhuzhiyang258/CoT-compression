#!/bin/bash

# 使用uv创建并管理Python环境
echo "Setting up CoT-compression environment with uv..."

# 安装uv (如果没有安装)
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source ~/.bashrc
fi

# 使用uv同步依赖
echo "Installing dependencies with uv..."
uv sync

echo "Environment setup complete!"
echo "Use 'uv run python run_inference.py' to run the inference script"