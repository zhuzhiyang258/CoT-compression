#!/usr/bin/env python3
"""
简单的模型调用脚本

支持传入系统提示词、用户提示词和调节各种模型参数
"""

import os
import sys
import argparse
import requests
import json
import time

def call_model(vllm_url: str,
               model_name: str,
               system_prompt: str,
               user_prompt: str,
               max_tokens: int = 8192,
               temperature: float = 0.8,
               top_p: float = 0.9,
               presence_penalty: float = 1.5,
               repetition_penalty: float = 1.05,
               timeout: int = 200) -> str:
    """
    调用vLLM模型服务
    
    参数:
        vllm_url: vLLM服务URL
        model_name: 模型名称
        system_prompt: 系统提示词
        user_prompt: 用户提示词
        max_tokens: 最大生成token数
        temperature: 温度参数
        top_p: top-p采样参数
        presence_penalty: 存在惩罚参数
        repetition_penalty: 重复惩罚参数
        timeout: 超时时间
    
    返回:
        模型响应文本
    """
    try:
        # 构建请求数据
        data = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "presence_penalty": presence_penalty,
            "repetition_penalty": repetition_penalty,
            "stream": False
        }
        
        # 发送请求
        print(f"正在调用模型: {model_name}")
        print(f"服务地址: {vllm_url}")
        
        start_time = time.time()
        response = requests.post(
            f"{vllm_url.rstrip('/')}/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            data=json.dumps(data),
            timeout=timeout
        )
        end_time = time.time()
        
        if response.status_code == 200:
            result = response.json()
            if 'choices' in result and len(result['choices']) > 0:
                content = result['choices'][0]['message']['content']
                print(f"调用成功 (耗时: {end_time - start_time:.2f}秒)")
                return content
            else:
                raise Exception("响应格式错误：未找到choices字段")
        else:
            raise Exception(f"API请求失败: {response.status_code} - {response.text}")
            
    except requests.exceptions.Timeout:
        raise Exception(f"请求超时 ({timeout}秒)")
    except requests.exceptions.ConnectionError:
        raise Exception(f"无法连接到服务器: {vllm_url}")
    except Exception as e:
        raise Exception(f"调用失败: {str(e)}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="简单的vLLM模型调用脚本")
    system_prompt = """
    你是金融会计专家。采用Chain of Draft方法：先草稿，后答案。\n\n【草稿规则】\n每个思维步骤写简洁草稿，每步最多5个词，只保留关键信息：计算结果、中间结论、关键判断。\n\n【推理结构】\n草稿1：题型识别（如：长投权益法）\n草稿2：关键数据（如：持股30%，净利600万）\n草稿3：计算方法（如：600×30%=180万）\n草稿4：验证结果（如：符合选项B,ps可能为单选也可能为多选，多选则用逗号分开）\n草稿5：确定答案（如：选择B,选择题答案可能是单选也可能是多选。）\n\n####\n最终答案：$\\boxed{答案}$
    """
    # 必需参数
    parser.add_argument("--system", type=str, default=system_prompt,
                       help="系统提示词")
    parser.add_argument("--user", type=str, required=True,
                       help="用户提示词")
    
    # 服务参数
    parser.add_argument("--url", type=str, default="http://localhost:6688",
                       help="vLLM服务器URL (默认: http://localhost:6688)")
    parser.add_argument("--model", type=str, default="Qwen3-4B",
                       help="模型名称 (默认: Qwen3-4B-Chat)")
    
    # 生成参数
    parser.add_argument("--max-tokens", type=int, default=4096,
                       help="最大生成token数 (默认: 1024)")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="温度参数，控制随机性 (默认: 0.7)")
    parser.add_argument("--top-p", type=float, default=0.9,
                       help="top-p采样参数 (默认: 0.9)")
    parser.add_argument("--presence-penalty", type=float, default=1.5,
                       help="存在惩罚参数 (默认: 1.5)")
    parser.add_argument("--repetition-penalty", type=float, default=1.05,
                       help="重复惩罚参数 (默认: 1.05)")
    parser.add_argument("--timeout", type=int, default=200,
                       help="请求超时时间，秒 (默认: 200)")
    
    # 输出选项
    parser.add_argument("--output", type=str, default=None,
                       help="保存结果到文件")
    parser.add_argument("--verbose", action="store_true",
                       help="显示详细信息")
    
    args = parser.parse_args()
    
    # 显示参数信息
    if args.verbose:
        print("=" * 50)
        print("调用参数:")
        print(f"  服务URL: {args.url}")
        print(f"  模型: {args.model}")
        print(f"  系统提示词: {args.system}")
        print(f"  用户提示词: {args.user}")
        print(f"  最大tokens: {args.max_tokens}")
        print(f"  温度: {args.temperature}")
        print(f"  top-p: {args.top_p}")
        print(f"  存在惩罚: {args.presence_penalty}")
        print(f"  重复惩罚: {args.repetition_penalty}")
        print(f"  超时: {args.timeout}秒")
        print("=" * 50)
    
    try:
        # 调用模型
        response = call_model(
            vllm_url=args.url,
            model_name=args.model,
            system_prompt=args.system,
            user_prompt=args.user,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            presence_penalty=args.presence_penalty,
            repetition_penalty=args.repetition_penalty,
            timeout=args.timeout
        )
        
        # 输出结果
        print("\n" + "=" * 50)
        print("模型响应:")
        print("=" * 50)
        print(response)
        print("=" * 50)
        
        # 保存到文件
        if args.output:
            try:
                with open(args.output, 'w', encoding='utf-8') as f:
                    f.write(response)
                print(f"\n结果已保存到: {args.output}")
            except Exception as e:
                print(f"\n保存文件失败: {e}")
                
    except Exception as e:
        print(f"\n错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 