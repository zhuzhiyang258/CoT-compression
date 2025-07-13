"""
vLLM并发版本主程序

此程序使用并发推理大幅提升推理效率，适用于批量处理大量数据。
"""

import os
import argparse
import time
import json
from data_loader import DataLoader
from model_inference_vllm_concurrent import VLLMConcurrentInference
from evaluator import Evaluator

def load_config(config_path):
    """
    加载配置文件。
    
    参数:
        config_path: 配置文件路径
    
    返回:
        配置字典，如果文件不存在或解析失败则返回None
    """
    try:
        if not os.path.exists(config_path):
            return None
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        return config
    except Exception as e:
        print(f"警告: 配置文件读取失败 ({config_path}): {e}")
        return None

def parse_args():
    """
    解析命令行参数。
    
    返回:
        解析后的参数
    """
    parser = argparse.ArgumentParser(description="金融领域长思维链压缩基线方法 (vLLM并发版本)")
    
    # 基本参数
    parser.add_argument("--input_path", type=str, default="./data/input.csv", help="输入数据路径")
    parser.add_argument("--output_path", type=str, default="./results_vllm_concurrent/output_vllm_concurrent.csv", help="输出数据路径")
    parser.add_argument("--reference_path", type=str, default=None, help="参考答案路径")
    parser.add_argument("--result_dir", type=str, default="./results_vllm_concurrent", help="结果保存目录")
    parser.add_argument("--config_path", type=str, default="./vllm_code/config.json", help="配置文件路径")
    
    # vLLM服务参数
    parser.add_argument("--vllm_url", type=str, default="http://localhost:6688", help="vLLM服务器URL")
    parser.add_argument("--model_name", type=str, default="Qwen3-4B", help="vLLM服务模型名称")
    parser.add_argument("--timeout", type=int, default=180, help="请求超时时间（秒）")
    parser.add_argument("--max_concurrent", type=int, default=20, help="最大并发数")
    
    # 生成参数
    parser.add_argument("--system_prompt", type=str, 
                       default="你是金融会计专家。采用Chain of Draft方法：先草稿，后答案。\n\n【草稿规则】\n每个思维步骤写简洁草稿，每步最多5个词，只保留关键信息：计算结果、中间结论、关键判断。\n\n【推理结构】\n草稿1：题型识别（如：长投权益法）\n草稿2：关键数据（如：持股30%，净利600万）\n草稿3：计算方法（如：600×30%=180万）\n草稿4：验证结果（如：符合选项B,ps可能为单选也可能为多选，多选则用逗号分开）\n草稿5：确定答案（如：选择B,选择题答案可能是单选也可能是多选。）\n\n####\n最终答案：$\\boxed{答案}$", 
                       help="系统提示词 (如果指定了prompt_type参数，此参数将被忽略)")
    parser.add_argument("--prompt_type", type=str, default=None, 
                       choices=["default", "detailed", "concise", "financial", "chain_of_thought"],
                       help="从配置文件中选择系统提示词类型")
    parser.add_argument("--max_new_tokens", type=int, default=4096, help="最大生成token数")
    parser.add_argument("--temperature", type=float, default=0.7, help="温度参数")
    parser.add_argument("--top_p", type=float, default=0.9, help="top-p采样参数")
    parser.add_argument("--num_samples", type=int, default=5, help="每个问题生成的样本数量")
    parser.add_argument("--max_questions", type=int, default=None, help="最大处理问题数量（用于测试）")
    parser.add_argument("--presence_penalty", type=float, default=1.5, help="存在惩罚参数 (默认: 1.5)")
    parser.add_argument("--repetition_penalty", type=float, default=1.05, help="重复惩罚参数 (默认: 1.05)")
    
    return parser.parse_args()

def main():
    """
    主函数。
    """
    # 解析命令行参数
    args = parse_args()
    
    # 加载配置文件
    config = load_config(args.config_path)
    if config:
        print(f"已加载配置文件: {args.config_path}")
        
        # 如果指定了prompt_type，从配置文件中读取对应的系统提示词
        if args.prompt_type and 'system_prompts' in config:
            if args.prompt_type in config['system_prompts']:
                args.system_prompt = config['system_prompts'][args.prompt_type]
                print(f"使用配置文件中的系统提示词类型: {args.prompt_type}")
            else:
                print(f"警告: 配置文件中未找到提示词类型 '{args.prompt_type}'，使用默认系统提示词")
        
        # 从配置文件中读取其他参数作为默认值（如果命令行未指定）
        if 'inference_settings' in config:
            inference_settings = config['inference_settings']
            # 只有当参数为默认值时才使用配置文件中的值
            if args.max_new_tokens == 8192 and 'max_new_tokens' in inference_settings:
                args.max_new_tokens = inference_settings['max_new_tokens']
            if args.temperature == 0.7 and 'temperature' in inference_settings:
                args.temperature = inference_settings['temperature']
            if args.top_p == 0.9 and 'top_p' in inference_settings:
                args.top_p = inference_settings['top_p']
            if args.num_samples == 5 and 'num_samples' in inference_settings:
                args.num_samples = inference_settings['num_samples']
            if args.presence_penalty == 1.5 and 'presence_penalty' in inference_settings:
                args.presence_penalty = inference_settings['presence_penalty']
            if args.repetition_penalty == 1.05 and 'repetition_penalty' in inference_settings:
                args.repetition_penalty = inference_settings['repetition_penalty']
        
        if 'concurrent_settings' in config:
            concurrent_settings = config['concurrent_settings']
            if args.max_concurrent == 20 and 'max_concurrent' in concurrent_settings:
                args.max_concurrent = concurrent_settings['max_concurrent']
            if args.timeout == 180 and 'timeout' in concurrent_settings:
                args.timeout = concurrent_settings['timeout']
    else:
        print("未找到配置文件，使用命令行参数和默认值")
    
    # 创建结果目录
    os.makedirs(args.result_dir, exist_ok=True)
    
    # 记录开始时间
    start_time = time.time()
    
    print("=== vLLM并发版本 CoT-compression 基线方法 ===")
    print(f"最大并发数: {args.max_concurrent}")
    
    # 加载数据
    print("\n=== 加载数据 ===")
    loader = DataLoader(args.input_path)
    questions = loader.get_questions()
    
    # 限制问题数量（用于测试）
    if args.max_questions is not None and args.max_questions > 0:
        questions = questions[:args.max_questions]
        print(f"限制处理前 {args.max_questions} 个问题（测试模式）")
    
    print(f"实际处理 {len(questions)} 个问题")
    
    # 准备prompts
    prompts = []
    for question in questions:
        prompts.append(loader.prepare_prompt(question, args.system_prompt))
    
    # 初始化vLLM并发模型推理
    print("\n=== 初始化vLLM并发模型推理 ===")
    inference = VLLMConcurrentInference(
        base_url=args.vllm_url,
        model_name=args.model_name,
        timeout=args.timeout,
        max_concurrent=args.max_concurrent
    )
    
    # 检查vLLM服务器连接
    try:
        inference.load_model()
    except Exception as e:
        print(f"无法连接到vLLM服务器: {e}")
        print("请确保vLLM服务器正在运行:")
        print("  启动命令: ./start_vllm_server.sh")
        print(f"  服务地址: {args.vllm_url}")
        return
    
    # 执行并发批量推理
    print("\n=== 执行并发模型推理 ===")
    print("推理参数:")
    print(f"  - 系统提示词: {args.system_prompt[:50]}{'...' if len(args.system_prompt) > 50 else ''}")
    if args.prompt_type:
        print(f"  - 提示词类型: {args.prompt_type}")
    print(f"  - 最大tokens: {args.max_new_tokens}")
    print(f"  - 温度: {args.temperature}")
    print(f"  - top_p: {args.top_p}")
    print(f"  - 每题样本数: {args.num_samples}")
    print(f"  - 存在惩罚: {args.presence_penalty}")
    print(f"  - 重复惩罚: {args.repetition_penalty}")
    print(f"  - 最大并发数: {args.max_concurrent}")
    print(f"  - 总任务数: {len(prompts)} × {args.num_samples} = {len(prompts) * args.num_samples}")
    
    # 估算时间
    estimated_time = (len(prompts) * args.num_samples * 15) / args.max_concurrent  # 假设每个任务15秒
    print(f"  - 预计耗时: {estimated_time/60:.1f}分钟")
    
    try:
        inference_start = time.time()
        results = inference.batch_inference_concurrent(
            prompts,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            num_samples=args.num_samples,
            presence_penalty=args.presence_penalty,
            repetition_penalty=args.repetition_penalty
        )
        inference_end = time.time()
        
        print(f"并发推理完成，共生成 {len(results)} 个样本")
        print(f"实际推理耗时: {(inference_end - inference_start)/60:.1f}分钟")
        
        # 统计成功和失败的数量
        successful = sum(1 for _, _, response in results if response.strip())
        failed = len(results) - successful
        print(f"成功: {successful}, 失败: {failed}")
        
    except Exception as e:
        print(f"推理过程中发生错误: {e}")
        return
    
    # 保存结果
    print("\n=== 保存结果 ===")
    loader.save_results(results, args.output_path)
    
    # 评估结果
    print("\n=== 评估结果 ===")
    if args.reference_path:
        try:
            evaluator = Evaluator(args.output_path, args.reference_path)
            report = evaluator.analyze_results(args.result_dir)
            print("评估完成！")
        except Exception as e:
            print(f"评估过程中发生错误: {e}")
    else:
        print("未提供参考答案，跳过评估")
    
    # 记录结束时间
    end_time = time.time()
    total_time = end_time - start_time
    print("\n=== 执行完成 ===")
    print(f"总耗时: {total_time/60:.1f}分钟 ({total_time:.2f}秒)")
    print(f"结果文件: {args.output_path}")
    print(f"结果目录: {args.result_dir}")
    
    # 性能统计
    total_tasks = len(prompts) * args.num_samples
    print("\n=== 性能统计 ===")
    print(f"总任务数: {total_tasks}")
    print(f"并发数: {args.max_concurrent}")
    print(f"吞吐量: {total_tasks/total_time:.2f} 任务/秒")
    print(f"平均每任务: {total_time/total_tasks:.2f} 秒")
    
    # 与串行版本的性能对比
    serial_estimated = total_tasks * 15  # 串行预计时间
    speedup = serial_estimated / total_time
    print(f"相比串行版本加速比: {speedup:.1f}x")

if __name__ == "__main__":
    main()