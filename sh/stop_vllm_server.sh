#!/bin/bash

# 停止vLLM服务器脚本

echo "=== 停止vLLM服务器 ==="

# 查找vLLM服务器进程
VLLM_PIDS=$(ps aux | grep "vllm serve" | grep -v grep | awk '{print $2}')

if [ -z "$VLLM_PIDS" ]; then
    echo "❌ 未找到运行中的vLLM服务器进程"
    exit 1
fi

echo "找到vLLM服务器进程："
ps aux | grep "vllm serve" | grep -v grep

# 尝试优雅停止
echo ""
echo "正在优雅停止vLLM服务器..."
for pid in $VLLM_PIDS; do
    echo "发送SIGTERM信号到进程 $pid"
    kill -TERM $pid
done

# 等待进程停止
echo "等待进程停止..."
sleep 5

# 检查进程是否已停止
REMAINING_PIDS=$(ps aux | grep "vllm serve" | grep -v grep | awk '{print $2}')

if [ -z "$REMAINING_PIDS" ]; then
    echo "✅ vLLM服务器已成功停止"
else
    echo "⚠️ 部分进程仍在运行，尝试强制停止..."
    
    # 强制停止
    for pid in $REMAINING_PIDS; do
        echo "发送SIGKILL信号到进程 $pid"
        kill -KILL $pid
    done
    
    sleep 2
    
    # 最终检查
    FINAL_CHECK=$(ps aux | grep "vllm serve" | grep -v grep | awk '{print $2}')
    
    if [ -z "$FINAL_CHECK" ]; then
        echo "✅ vLLM服务器已强制停止"
    else
        echo "❌ 无法停止部分vLLM进程，请手动处理："
        ps aux | grep "vllm serve" | grep -v grep
        exit 1
    fi
fi

# 检查端口占用
echo ""
echo "检查端口6688占用情况..."
PORT_USAGE=$(lsof -ti:6688 2>/dev/null)

if [ -n "$PORT_USAGE" ]; then
    echo "⚠️ 端口6688仍被占用，进程ID: $PORT_USAGE"
    echo "是否要停止占用端口的进程? (y/N)"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        kill -TERM $PORT_USAGE
        sleep 2
        echo "✅ 已停止占用端口6688的进程"
    fi
else
    echo "✅ 端口6688已释放"
fi

# 清理日志文件（可选）
if [ -f "vllm_server.log" ]; then
    echo ""
    echo "是否要清理vLLM服务器日志? (y/N)"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        mv vllm_server.log "vllm_server_$(date +%Y%m%d_%H%M%S).log"
        echo "✅ 日志文件已备份并清理"
    fi
fi

echo ""
echo "=== vLLM服务器停止完成 ==="