#!/usr/bin/env python3
"""
SFT Training Script with LoRA for Qwen3-4B-Chat
"""

import os
import sys
import json
import yaml
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    set_seed
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset, load_dataset
import transformers

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load training configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def load_training_data(data_dir: str) -> Dataset:
    """Load training data from jsonl files."""
    data_files = []
    data_path = Path(data_dir)
    
    # Look for jsonl files in the data directory
    for file_path in data_path.glob("*.jsonl"):
        data_files.append(str(file_path))
    
    if not data_files:
        # If no jsonl files, look for the training dataset in parent directory
        parent_data_path = data_path.parent / "data" / "training_dataset.jsonl"
        if parent_data_path.exists():
            data_files.append(str(parent_data_path))
    
    if not data_files:
        raise FileNotFoundError(f"No training data found in {data_dir}")
    
    logger.info(f"Loading training data from: {data_files}")
    dataset = load_dataset("json", data_files=data_files, split="train")
    
    return dataset


def preprocess_function(examples, tokenizer, cutoff_len: int):
    """Preprocess training examples for Qwen format."""
    input_ids_list = []
    attention_mask_list = []
    labels_list = []
    
    for i in range(len(examples["instruction"])):
        instruction = examples["instruction"][i]
        input_text = examples.get("input", [""])[i] if "input" in examples else ""
        output = examples["output"][i]
        
        # Format prompt according to Qwen template
        if input_text and input_text.strip():
            prompt = f"<|im_start|>user\n{instruction}\n{input_text}<|im_end|>\n<|im_start|>assistant\n"
        else:
            prompt = f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
        
        full_text = prompt + output + "<|im_end|>"
        
        # Tokenize full text
        tokenized = tokenizer(
            full_text,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        
        # Tokenize prompt only to get length
        prompt_tokenized = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        
        # Create labels - mask prompt tokens with -100
        input_ids = tokenized["input_ids"]
        labels = input_ids.copy()
        prompt_len = len(prompt_tokenized["input_ids"])
        
        # Mask prompt tokens
        for j in range(min(prompt_len, len(labels))):
            labels[j] = -100
        
        input_ids_list.append(input_ids)
        attention_mask_list.append(tokenized["attention_mask"])
        labels_list.append(labels)
    
    return {
        "input_ids": input_ids_list,
        "attention_mask": attention_mask_list,
        "labels": labels_list,
    }


def setup_model_and_tokenizer(config: Dict[str, Any]):
    """Setup model and tokenizer with LoRA configuration."""
    model_name = config["model_name_or_path"]
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="right",
        model_max_length=config["cutoff_len"],
    )
    
    # Add pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=getattr(torch, config["torch_dtype"]),
        device_map="auto",
        trust_remote_code=True,
        attn_implementation=config.get("attn_implementation", "eager"),
    )
    
    # Prepare model for LoRA training (only if using quantization)
    # model = prepare_model_for_kbit_training(model)
    
    # Setup LoRA configuration
    lora_config = LoraConfig(
        r=config["lora_rank"],
        lora_alpha=config["lora_alpha"],
        target_modules=config["lora_target_modules"],
        lora_dropout=config["lora_dropout"],
        bias=config.get("lora_bias", "none"),
        task_type="CAUSAL_LM",
        use_rslora=config.get("use_rslora", False),
        use_dora=config.get("use_dora", False),
    )
    
    # Apply LoRA to model
    model = get_peft_model(model, lora_config)
    
    # Enable training mode
    model.train()
    
    # Ensure LoRA parameters require gradients
    for name, param in model.named_parameters():
        if 'lora_' in name:
            param.requires_grad = True
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="SFT Training with LoRA")
    parser.add_argument(
        "--config", 
        type=str, 
        default="./sft/configs/lora_config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./sft/data",
        help="Directory containing training data"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./sft/output",
        help="Output directory for model checkpoints"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override with command line arguments
    if args.data_dir:
        config["dataset_dir"] = args.data_dir
    if args.output_dir:
        config["output_dir"] = args.output_dir
    
    # Set seed for reproducibility
    set_seed(config["seed"])
    
    # Setup model and tokenizer
    logger.info("Loading model and tokenizer...")
    model, tokenizer = setup_model_and_tokenizer(config)
    
    # Load training data
    logger.info("Loading training data...")
    train_dataset = load_training_data(config["dataset_dir"])
    
    # Preprocess data
    logger.info("Preprocessing data...")
    train_dataset = train_dataset.map(
        lambda examples: preprocess_function(examples, tokenizer, config["cutoff_len"]),
        batched=True,
        num_proc=config.get("preprocessing_num_workers", 4),
        remove_columns=train_dataset.column_names,
    )
    
    # Setup data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
    )
    
    # Setup training arguments
    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        overwrite_output_dir=config["overwrite_output_dir"],
        num_train_epochs=config["num_train_epochs"],
        max_steps=config.get("max_steps", -1),  # Support max_steps for testing
        per_device_train_batch_size=config["per_device_train_batch_size"],
        per_device_eval_batch_size=config["per_device_eval_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        lr_scheduler_type=config["lr_scheduler_type"],
        warmup_ratio=config["warmup_ratio"],
        fp16=config["fp16"],
        bf16=config["bf16"],
        gradient_checkpointing=config["gradient_checkpointing"],
        dataloader_pin_memory=config["dataloader_pin_memory"],
        remove_unused_columns=config["remove_unused_columns"],
        optim=config["optim"],
        group_by_length=config["group_by_length"],
        logging_steps=config["logging_steps"],
        save_strategy=config["save_strategy"],
        save_steps=config["save_steps"],
        eval_strategy=config.get("evaluation_strategy", "no"),
        save_total_limit=config["save_total_limit"],
        seed=config["seed"],
        dataloader_num_workers=config["dataloader_num_workers"],
        report_to=config.get("report_to", []),
        ddp_timeout=config.get("ddp_timeout", 1800),
        include_num_input_tokens_seen=config.get("include_num_input_tokens_seen", True),
    )
    
    # Setup trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Start training
    logger.info("Starting training...")
    trainer.train()
    
    # Save final model
    logger.info("Saving final model...")
    trainer.save_model()
    trainer.save_state()
    
    # Save LoRA adapters separately
    model.save_pretrained(os.path.join(config["output_dir"], "lora_adapters"))
    tokenizer.save_pretrained(os.path.join(config["output_dir"], "lora_adapters"))
    
    logger.info("Training completed!")


if __name__ == "__main__":
    main()