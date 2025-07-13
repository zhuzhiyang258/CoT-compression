#!/usr/bin/env python3
"""
CoT-compression 多线程推理主程序
使用uv管理的简化启动脚本，调用核心多线程推理功能
"""

import os
import sys

# 添加src路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# 导入并运行主程序
from scripts.main_vllm_concurrent import main

if __name__ == "__main__":
    main()