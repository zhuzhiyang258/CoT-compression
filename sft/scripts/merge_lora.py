#!/usr/bin/env python3
"""
LoRA权重合并脚本
将LoRA适配器权重与基础模型合并，生成完整的微调模型
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def load_and_merge_model(
    base_model_path: str,
    lora_adapter_path: str,
    output_path: str,
    torch_dtype: str = "bfloat16",
    device_map: str = "auto"
):
    """
    加载基础模型和LoRA适配器，合并后保存完整模型
    
    Args:
        base_model_path: 基础模型路径
        lora_adapter_path: LoRA适配器路径
        output_path: 输出模型路径
        torch_dtype: 模型精度类型
        device_map: 设备映射策略
    """
    logger.info(f"Loading base model from: {base_model_path}")
    
    # 加载基础模型
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=getattr(torch, torch_dtype),
        device_map=device_map,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    
    logger.info(f"Loading LoRA adapter from: {lora_adapter_path}")
    
    # 加载LoRA配置
    peft_config = PeftConfig.from_pretrained(lora_adapter_path)
    
    # 加载LoRA模型
    model = PeftModel.from_pretrained(
        base_model,
        lora_adapter_path,
        torch_dtype=getattr(torch, torch_dtype),
    )
    
    logger.info("Merging LoRA weights with base model...")
    
    # 合并LoRA权重到基础模型
    merged_model = model.merge_and_unload()
    
    # 确保输出目录存在
    os.makedirs(output_path, exist_ok=True)
    
    logger.info(f"Saving merged model to: {output_path}")
    
    # 保存合并后的模型
    merged_model.save_pretrained(
        output_path,
        safe_serialization=True,
        max_shard_size="5GB"
    )
    
    # 加载并保存tokenizer
    logger.info("Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True
    )
    tokenizer.save_pretrained(output_path)
    
    logger.info("✅ Model merge completed successfully!")
    
    return merged_model, tokenizer


def validate_merged_model(model_path: str, test_prompt: str = "Hello, how are you?"):
    """
    验证合并后的模型是否可以正常推理
    
    Args:
        model_path: 合并后模型的路径
        test_prompt: 测试提示词
    """
    logger.info(f"Validating merged model at: {model_path}")
    
    try:
        # 加载合并后的模型和tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # 设置pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # 测试推理
        logger.info(f"Testing inference with prompt: '{test_prompt}'")
        
        inputs = tokenizer(test_prompt, return_tensors="pt", padding=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # 解码输出
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"Model response: {response}")
        
        logger.info("✅ Model validation successful!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Model validation failed: {e}")
        return False


def get_model_info(model_path: str):
    """
    获取模型信息
    
    Args:
        model_path: 模型路径
    """
    try:
        from transformers import AutoConfig
        
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        
        info = {
            "model_type": getattr(config, "model_type", "unknown"),
            "hidden_size": getattr(config, "hidden_size", "unknown"),
            "num_attention_heads": getattr(config, "num_attention_heads", "unknown"),
            "num_hidden_layers": getattr(config, "num_hidden_layers", "unknown"),
            "vocab_size": getattr(config, "vocab_size", "unknown"),
        }
        
        logger.info("📋 Model Information:")
        for key, value in info.items():
            logger.info(f"  {key}: {value}")
            
        return info
        
    except Exception as e:
        logger.warning(f"Failed to get model info: {e}")
        return {}


def calculate_model_size(model_path: str):
    """
    计算模型文件大小
    
    Args:
        model_path: 模型路径
    """
    try:
        total_size = 0
        for root, dirs, files in os.walk(model_path):
            for file in files:
                if file.endswith(('.safetensors', '.bin', '.pt', '.pth')):
                    file_path = os.path.join(root, file)
                    total_size += os.path.getsize(file_path)
        
        # Convert to human readable format
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if total_size < 1024.0:
                size_str = f"{total_size:.2f} {unit}"
                break
            total_size /= 1024.0
        else:
            size_str = f"{total_size:.2f} PB"
        
        logger.info(f"📦 Model size: {size_str}")
        return size_str
        
    except Exception as e:
        logger.warning(f"Failed to calculate model size: {e}")
        return "unknown"


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter with base model")
    parser.add_argument(
        "--base_model",
        type=str,
        default="./models/Qwen3-4B-Chat",
        help="Path to base model"
    )
    parser.add_argument(
        "--lora_adapter",
        type=str,
        required=True,
        help="Path to LoRA adapter (e.g., ./sft/test_output/lora_adapters)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for merged model"
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Torch dtype for model"
    )
    parser.add_argument(
        "--device_map",
        type=str,
        default="auto",
        help="Device map for model loading"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate merged model after creation"
    )
    parser.add_argument(
        "--test_prompt",
        type=str,
        default="你好，请介绍一下自己。",
        help="Test prompt for model validation"
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Remove intermediate files after merge"
    )
    
    args = parser.parse_args()
    
    # 验证输入路径
    if not os.path.exists(args.base_model):
        logger.error(f"Base model path not found: {args.base_model}")
        return 1
    
    if not os.path.exists(args.lora_adapter):
        logger.error(f"LoRA adapter path not found: {args.lora_adapter}")
        return 1
    
    # 检查LoRA适配器文件
    adapter_config_path = os.path.join(args.lora_adapter, "adapter_config.json")
    if not os.path.exists(adapter_config_path):
        logger.error(f"adapter_config.json not found in: {args.lora_adapter}")
        return 1
    
    try:
        logger.info("🚀 Starting LoRA merge process...")
        logger.info(f"Base model: {args.base_model}")
        logger.info(f"LoRA adapter: {args.lora_adapter}")
        logger.info(f"Output directory: {args.output_dir}")
        
        # 执行合并
        merged_model, tokenizer = load_and_merge_model(
            base_model_path=args.base_model,
            lora_adapter_path=args.lora_adapter,
            output_path=args.output_dir,
            torch_dtype=args.torch_dtype,
            device_map=args.device_map
        )
        
        # 获取模型信息
        get_model_info(args.output_dir)
        calculate_model_size(args.output_dir)
        
        # 验证模型
        if args.validate:
            logger.info("🔍 Validating merged model...")
            validation_success = validate_merged_model(
                args.output_dir,
                args.test_prompt
            )
            if not validation_success:
                logger.warning("Model validation failed, but merge may still be successful")
        
        # 清理中间文件
        if args.cleanup:
            logger.info("🧹 Cleaning up intermediate files...")
            # 这里可以添加清理逻辑，比如删除临时文件等
        
        logger.info("✅ LoRA merge process completed successfully!")
        logger.info(f"Merged model saved to: {args.output_dir}")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ LoRA merge failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())