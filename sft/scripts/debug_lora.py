#!/usr/bin/env python3
"""
Debug script to test LoRA setup
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

def test_lora_setup():
    print("🔧 Testing LoRA setup...")
    
    # Load model and tokenizer
    model_path = "./models/Qwen3-4B-Chat"
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side="right",
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    print("Setting up LoRA...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config)
    model.train()
    
    print("Model parameters:")
    for name, param in model.named_parameters():
        print(f"  {name}: requires_grad={param.requires_grad}")
        if 'lora_' in name:
            break
    
    model.print_trainable_parameters()
    
    # Test forward pass
    print("\nTesting forward pass...")
    test_input = tokenizer("Hello world", return_tensors="pt", padding=True)
    test_input = {k: v.to(model.device) for k, v in test_input.items()}
    
    with torch.no_grad():
        outputs = model(**test_input)
        print(f"Forward pass successful! Loss shape: {outputs.logits.shape}")
    
    # Test with labels for training
    print("\nTesting training step...")
    test_input["labels"] = test_input["input_ids"].clone()
    
    outputs = model(**test_input)
    loss = outputs.loss
    print(f"Training loss: {loss}")
    
    # Test backward
    loss.backward()
    print("✅ Backward pass successful!")
    
    # Check gradients
    has_grad = False
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"  {name}: has gradient")
            has_grad = True
            break
    
    if has_grad:
        print("✅ Gradients computed successfully!")
    else:
        print("❌ No gradients found!")

if __name__ == "__main__":
    test_lora_setup()