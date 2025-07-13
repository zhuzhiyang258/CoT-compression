#!/usr/bin/env python3
"""
快速LoRA合并脚本 - 用于测试和验证
"""

import os
import argparse
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def quick_merge(lora_path: str, output_path: str = None):
    """快速合并LoRA权重"""
    
    if output_path is None:
        output_path = f"{lora_path}_merged"
    
    logger.info(f"🔗 快速LoRA合并")
    logger.info(f"LoRA路径: {lora_path}")
    logger.info(f"输出路径: {output_path}")
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import PeftModel, PeftConfig
        import torch
        
        # 检查LoRA适配器
        if not os.path.exists(lora_path):
            raise FileNotFoundError(f"LoRA适配器路径不存在: {lora_path}")
        
        adapter_config_path = os.path.join(lora_path, "adapter_config.json")
        if not os.path.exists(adapter_config_path):
            raise FileNotFoundError(f"adapter_config.json不存在: {adapter_config_path}")
        
        # 读取配置获取基础模型路径
        peft_config = PeftConfig.from_pretrained(lora_path)
        base_model_path = peft_config.base_model_name_or_path
        
        # 如果是相对路径，尝试本地路径
        if not os.path.exists(base_model_path):
            local_base_path = "./models/Qwen3-4B-Chat"
            if os.path.exists(local_base_path):
                base_model_path = local_base_path
                logger.info(f"使用本地基础模型: {base_model_path}")
            else:
                raise FileNotFoundError(f"基础模型未找到: {base_model_path}")
        
        logger.info(f"基础模型: {base_model_path}")
        
        # 加载基础模型
        logger.info("加载基础模型...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        
        # 加载LoRA模型
        logger.info("加载LoRA适配器...")
        model = PeftModel.from_pretrained(
            base_model,
            lora_path,
            torch_dtype=torch.bfloat16,
        )
        
        # 合并权重
        logger.info("合并权重...")
        merged_model = model.merge_and_unload()
        
        # 保存合并后的模型
        logger.info("保存合并后的模型...")
        os.makedirs(output_path, exist_ok=True)
        merged_model.save_pretrained(
            output_path,
            safe_serialization=True,
            max_shard_size="5GB"
        )
        
        # 保存tokenizer
        logger.info("保存tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            trust_remote_code=True
        )
        tokenizer.save_pretrained(output_path)
        
        logger.info("✅ 合并完成!")
        
        # 简单测试
        logger.info("🧪 快速测试...")
        test_input = "你好，请介绍一下自己。"
        inputs = tokenizer(test_input, return_tensors="pt")
        inputs = {k: v.to(merged_model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = merged_model.generate(
                **inputs,
                max_new_tokens=30,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"测试输出: {response}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 合并失败: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="快速LoRA合并")
    parser.add_argument("lora_path", help="LoRA适配器路径")
    parser.add_argument("-o", "--output", help="输出路径 (默认: {lora_path}_merged)")
    
    args = parser.parse_args()
    
    success = quick_merge(args.lora_path, args.output)
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())