# CoT-compression

财经领域长思维链压缩基线实现 

**目标**: 在保持答案准确性的前提下，通过压缩推理过程来减少token使用量，实现高效的思维链推理。

## 评分规则

1. **准确性**: 方法通过的最低标准为准确率90%。只要5条回复中存在1条答对，则认为模型正确回答该问题。
2. **高效性**: 对每个问题取正确回答sample中的最短模型原始回复长度作为该问题的cot长度。
3. **最终得分**: 所有问题cot长度的总和的相反数。也就是说，cot长度越短，得分越高。

## 项目结构

```
CoT-compression/
├── src/
│   ├── core/           # 核心数据处理模块
│   │   └── data_loader.py
│   ├── inference/      # 模型推理模块
│   │   └── model_inference_vllm_concurrent.py
│   ├── evaluation/     # 评估模块
│   │   └── evaluator.py
│   └── scripts/        # 主要脚本
│       └── main_vllm_concurrent.py
├── config/             # 配置文件
│   ├── start_vllm_server.sh
│   └── stop_vllm_server.sh
├── data/               # 数据文件
├── models/             # 模型文件 (Qwen3-4B-Base)
├── docs/               # 文档
├── pyproject.toml      # uv项目配置
└── run_inference.py    # 主要启动脚本
```

## 快速开始

### 1. 环境设置 (使用uv)
```bash
# 安装uv (如果未安装)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 设置项目环境
bash setup_env.sh
```

### 2. 启动vLLM服务器
```bash
bash config/start_vllm_server.sh
```

### 3. 运行多线程推理
```bash
# 使用uv运行
uv run python run_inference.py

# 或者直接运行 (如果已激活uv环境)
python run_inference.py
```

### 4. 停止vLLM服务器
```bash
bash config/stop_vllm_server.sh
```

## 输出格式

输出CSV文件包含4列：
- `total_id`: 总编号 (Overall sequential ID)
- `question_id`: 问题编号 (Question ID)  
- `sample_id`: 样本编号 (Sample ID for that question)
- `model_output`: 模型原始输出 (Raw model output including reasoning and \boxed{answer})

## 主要参数

- `--input_path`: 输入数据路径 (默认: ./data/input.csv)
- `--output_path`: 输出数据路径 (默认: ./output.csv)
- `--max_concurrent`: 最大并发数 (默认: 8)
- `--num_samples`: 每题样本数 (默认: 5)
- `--temperature`: 温度参数 (默认: 0.3)
- `--model_name`: 模型名称 (默认: Qwen3-4B-Base)

## 思维链压缩策略

当前使用的压缩提示词：
```
Think step by step, but only keep a minimum draft for each thinking step, with 5 words at most. Return the answer at the end of the response after a separator ####.
```

目标是在维持90%准确率的同时，最小化推理过程的token使用量。