#!/usr/bin/env python3
"""
使用config.json配置运行CoT压缩推理
"""

import json
import os
import sys
import argparse
import time

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.core.data_loader import DataLoader
from src.inference.model_inference_vllm_concurrent import VLLMConcurrentInference

def load_config(config_path="./vllm_code/config.json"):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def list_available_prompts(config):
    """列出所有可用的提示词"""
    print("可用的提示词版本:")
    print("=" * 60)
    
    prompts = config["system_prompts"]
    descriptions = config.get("prompt_descriptions", {})
    
    for name, prompt in prompts.items():
        if name == "default":
            continue
            
        print(f"\n【{name.upper()}】")
        if name in descriptions:
            print(f"描述: {descriptions[name]}")
        print(f"长度: {len(prompt)} 字符")
        print(f"预览: {prompt[:80]}...")
        print("-" * 50)
    
    print(f"\n默认版本: {prompts.get('default', 'smart_reasoning')}")
    print(f"推荐版本: {config['optimization_notes']['recommended_usage']['primary']}")

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="基于config.json的CoT压缩推理")
    
    parser.add_argument("--prompt_version", type=str, default="default",
                       help="选择提示词版本")
    parser.add_argument("--config_path", type=str, default="./vllm_code/config.json",
                       help="配置文件路径")
    parser.add_argument("--list_prompts", action="store_true",
                       help="列出所有可用提示词")
    parser.add_argument("--input_path", type=str, default="./data/input.csv",
                       help="输入数据路径")
    parser.add_argument("--output_path", type=str, default="./output_config.csv",
                       help="输出数据路径")
    parser.add_argument("--vllm_url", type=str, default="http://localhost:6688",
                       help="vLLM服务器URL")
    parser.add_argument("--max_questions", type=int, default=None,
                       help="最大处理问题数量（测试用）")
    parser.add_argument("--presence_penalty", type=float, default=1.5,
                       help="存在惩罚参数 (默认: 1.5)")
    parser.add_argument("--repetition_penalty", type=float, default=1.05,
                       help="重复惩罚参数 (默认: 1.05)")
    
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_args()
    
    # 加载配置
    try:
        config = load_config(args.config_path)
        print(f"✓ 成功加载配置文件: {args.config_path}")
    except Exception as e:
        print(f"✗ 无法加载配置文件: {e}")
        return
    
    # 如果只是列出提示词，直接返回
    if args.list_prompts:
        list_available_prompts(config)
        return
    
    # 获取指定的提示词
    system_prompts = config["system_prompts"]
    if args.prompt_version not in system_prompts:
        print(f"✗ 提示词版本 '{args.prompt_version}' 不存在")
        print(f"可用版本: {list(system_prompts.keys())}")
        return
    
    # 如果指定的是default，获取实际的默认版本
    if args.prompt_version == "default":
        actual_version = system_prompts["default"]
        system_prompt = system_prompts[actual_version]
        print(f"使用默认版本: {actual_version}")
    else:
        system_prompt = system_prompts[args.prompt_version]
        actual_version = args.prompt_version
    
    # 获取推理设置
    inference_settings = config["inference_settings"]
    concurrent_settings = config["concurrent_settings"]
    
    print(f"\n{'='*60}")
    print("CoT压缩推理 - 配置文件模式")
    print(f"{'='*60}")
    print(f"提示词版本: {actual_version}")
    print(f"提示词长度: {len(system_prompt)} 字符")
    print(f"max_new_tokens: {inference_settings['max_new_tokens']}")
    print(f"temperature: {inference_settings['temperature']}")
    print(f"top_p: {inference_settings['top_p']}")
    print(f"num_samples: {inference_settings['num_samples']}")
    print(f"presence_penalty: {inference_settings.get('presence_penalty', args.presence_penalty)}")
    print(f"repetition_penalty: {inference_settings.get('repetition_penalty', args.repetition_penalty)}")
    print(f"max_concurrent: {concurrent_settings['max_concurrent']}")
    
    # 加载数据
    print("\n=== 加载数据 ===")
    loader = DataLoader(args.input_path)
    questions = loader.get_questions()
    
    # 限制问题数量（测试用）
    if args.max_questions is not None and args.max_questions > 0:
        questions = questions[:args.max_questions]
        print(f"限制处理前 {args.max_questions} 个问题（测试模式）")
    
    print(f"实际处理 {len(questions)} 个问题")
    
    # 准备prompts
    prompts = []
    for question in questions:
        # 清理问题文本
        if "问题：\\n" in question:
            clean_question = question[question.find("问题：\\n") + 4:].strip()
        else:
            clean_question = question
            
        prompts.append({
            "system": system_prompt,
            "user": clean_question
        })
    
    # 初始化推理器
    print("\n=== 初始化vLLM推理器 ===")
    try:
        inference = VLLMConcurrentInference(
            base_url=args.vllm_url,
            max_concurrent=concurrent_settings["max_concurrent"],
            timeout=concurrent_settings["timeout"]
        )
        inference.load_model()
        print(f"✓ 成功连接vLLM服务器: {args.vllm_url}")
    except Exception as e:
        print(f"✗ 无法连接vLLM服务器: {e}")
        print("请确保服务器正在运行: bash sh/start_vllm_server.sh")
        return
    
    # 执行推理
    print("\n=== 执行推理 ===")
    print(f"总任务数: {len(prompts)} × {inference_settings['num_samples']} = {len(prompts) * inference_settings['num_samples']}")
    
    start_time = time.time()
    try:
        results = inference.batch_inference_concurrent(
            prompts,
            max_new_tokens=inference_settings["max_new_tokens"],
            temperature=inference_settings["temperature"],
            top_p=inference_settings["top_p"],
            num_samples=inference_settings["num_samples"],
            presence_penalty=inference_settings.get("presence_penalty", args.presence_penalty),
            repetition_penalty=inference_settings.get("repetition_penalty", args.repetition_penalty)
        )
        
        end_time = time.time()
        print(f"\n✓ 推理完成，共生成 {len(results)} 个样本")
        print(f"耗时: {(end_time - start_time)/60:.1f} 分钟")
        
        # 分析结果质量
        successful = sum(1 for _, _, _, response in results if response.strip())
        format_correct = sum(1 for _, _, _, response in results 
                           if "$\\boxed{" in response)
        think_tags = sum(1 for _, _, _, response in results 
                        if "<think>" in response and "</think>" in response)
        
        print(f"成功生成: {successful}/{len(results)} ({successful/len(results)*100:.1f}%)")
        print(f"格式正确: {format_correct}/{len(results)} ({format_correct/len(results)*100:.1f}%)")
        print(f"包含推理: {think_tags}/{len(results)} ({think_tags/len(results)*100:.1f}%)")
        
        # 计算平均响应长度
        response_lengths = [len(response) for _, _, _, response in results if response.strip()]
        if response_lengths:
            avg_length = sum(response_lengths) / len(response_lengths)
            print(f"平均响应长度: {avg_length:.0f} 字符")
        
    except Exception as e:
        print(f"✗ 推理过程出错: {e}")
        return
    
    # 保存结果
    print("\n=== 保存结果 ===")
    loader.save_results(results, args.output_path)
    
    # 输出总结
    print(f"\n{'='*60}")
    print("执行完成")
    print(f"{'='*60}")
    print(f"使用配置: {args.config_path}")
    print(f"提示词版本: {actual_version}")
    print(f"输出文件: {args.output_path}")
    print(f"处理问题数: {len(questions)}")
    print(f"生成样本数: {len(results)}")
    
    # 显示优化效果
    if "optimization_notes" in config:
        print("\n关键改进:")
        for improvement in config["optimization_notes"]["key_improvements"]:
            print(f"  - {improvement}")

if __name__ == "__main__":
    main()