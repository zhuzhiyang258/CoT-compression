#!/bin/bash

# 清理旧的输出文件脚本
# 用法: bash sft/scripts/cleanup_old_outputs.sh [--dry-run]

set -e

echo "🧹 SFT输出文件清理工具"
echo "============================================"

DRY_RUN=false
if [ "$1" = "--dry-run" ]; then
    DRY_RUN=true
    echo "🔍 预览模式 - 不会实际删除文件"
fi

# 定义要清理的旧输出目录
OLD_DIRS=(
    "sft/test_output"
    "sft/financial_output" 
    "sft/financial_merged_model"
    "sft/merged_model"
    "sft/quick_merged_model"
    "sft/full_merged_model"
)

echo ""
echo "📊 当前输出目录大小:"
total_size=0
for dir in "${OLD_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        size=$(du -sh "$dir" 2>/dev/null | cut -f1 || echo "0")
        echo "  $dir: $size"
        # Convert size to GB for rough calculation
        size_gb=$(du -s "$dir" 2>/dev/null | awk '{print $1/1024/1024}' || echo "0")
        total_size=$(echo "$total_size + $size_gb" | bc -l 2>/dev/null || echo "$total_size")
    fi
done

echo ""
printf "📦 预计可释放空间: %.1fGB\n" "$total_size" 2>/dev/null || echo "📦 预计可释放空间: ~15GB"

echo ""
echo "🆕 新的统一输出结构已配置:"
echo "  sft/outputs/training/     - 训练输出"
echo "  sft/outputs/merged_models/ - 合并模型"

echo ""
if [ "$DRY_RUN" = true ]; then
    echo "将要删除的目录 (预览):"
    for dir in "${OLD_DIRS[@]}"; do
        if [ -d "$dir" ]; then
            echo "  ❌ $dir"
        fi
    done
    echo ""
    echo "要实际执行清理，请运行:"
    echo "  bash sft/scripts/cleanup_old_outputs.sh"
else
    echo "⚠️  即将删除旧的输出目录，这个操作不可逆！"
    echo "按 Ctrl+C 取消，或按任意键继续..."
    read -n 1 -s
    
    echo ""
    echo "开始清理..."
    
    for dir in "${OLD_DIRS[@]}"; do
        if [ -d "$dir" ]; then
            echo "🗑️  删除: $dir"
            rm -rf "$dir"
        fi
    done
    
    echo ""
    echo "✅ 清理完成！"
    echo ""
    echo "📁 新的输出目录结构:"
    echo "  sft/outputs/ - 统一输出目录 (已配置到gitignore)"
    echo ""
    echo "🚀 后续训练将使用新的输出结构"
fi