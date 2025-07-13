#!/usr/bin/env python3
"""
Data preparation script for SFT training
Converts various data formats to the required JSONL format for training
"""

import json
import csv
import argparse
from pathlib import Path
from typing import Dict, Any, List


def convert_csv_to_jsonl(csv_path: str, output_path: str, instruction_col: str = "question_text", 
                        output_col: str = "answer", input_col: str = None):
    """Convert CSV to JSONL format for training."""
    data = []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            entry = {
                "instruction": row[instruction_col],
                "output": row[output_col] if output_col in row else "Please provide the answer.",
            }
            
            if input_col and input_col in row and row[input_col].strip():
                entry["input"] = row[input_col]
            else:
                entry["input"] = ""
                
            data.append(entry)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"Converted {len(data)} entries from {csv_path} to {output_path}")


def convert_existing_dataset(input_path: str, output_path: str):
    """Convert existing training dataset to SFT format."""
    data = []
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                
                # Handle different input formats
                if "question_text" in item:
                    instruction = item["question_text"]
                elif "instruction" in item:
                    instruction = item["instruction"]
                elif "question" in item:
                    instruction = item["question"]
                else:
                    continue
                
                # Extract answer/output
                if "answer" in item:
                    output = item["answer"]
                elif "output" in item:
                    output = item["output"]
                elif "response" in item:
                    output = item["response"]
                else:
                    output = "Please provide the answer."
                
                entry = {
                    "instruction": instruction,
                    "input": "",
                    "output": output
                }
                data.append(entry)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"Converted {len(data)} entries from {input_path} to {output_path}")


def create_sample_data(output_path: str, num_samples: int = 10):
    """Create sample training data for testing."""
    sample_data = [
        {
            "instruction": "What is the capital of France?",
            "input": "",
            "output": "The capital of France is Paris."
        },
        {
            "instruction": "Calculate the sum of 15 and 27.",
            "input": "",
            "output": "The sum of 15 and 27 is 42."
        },
        {
            "instruction": "Explain the concept of machine learning.",
            "input": "",
            "output": "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed for every task."
        },
        {
            "instruction": "Solve this math problem:",
            "input": "If a train travels 120 km in 2 hours, what is its average speed?",
            "output": "The average speed is 120 km ÷ 2 hours = 60 km/h."
        },
        {
            "instruction": "What is the largest planet in our solar system?",
            "input": "",
            "output": "Jupiter is the largest planet in our solar system."
        }
    ]
    
    # Repeat and vary the sample data
    data = []
    for i in range(num_samples):
        entry = sample_data[i % len(sample_data)].copy()
        if i >= len(sample_data):
            entry["instruction"] = f"[Sample {i+1}] " + entry["instruction"]
        data.append(entry)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"Created {len(data)} sample entries in {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare data for SFT training")
    parser.add_argument("--input", type=str, help="Input data file path")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL file path")
    parser.add_argument("--format", type=str, choices=["csv", "jsonl", "sample"], 
                       required=True, help="Input data format")
    parser.add_argument("--instruction_col", type=str, default="question_text",
                       help="Column name for instructions (CSV format)")
    parser.add_argument("--output_col", type=str, default="answer",
                       help="Column name for outputs (CSV format)")
    parser.add_argument("--input_col", type=str, default=None,
                       help="Column name for additional input (CSV format)")
    parser.add_argument("--num_samples", type=int, default=10,
                       help="Number of sample entries to create (sample format)")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if args.format == "csv":
        if not args.input:
            raise ValueError("Input file is required for CSV format")
        convert_csv_to_jsonl(args.input, args.output, args.instruction_col, 
                           args.output_col, args.input_col)
    elif args.format == "jsonl":
        if not args.input:
            raise ValueError("Input file is required for JSONL format")
        convert_existing_dataset(args.input, args.output)
    elif args.format == "sample":
        create_sample_data(args.output, args.num_samples)
    
    print(f"Data preparation completed! Output saved to: {args.output}")


if __name__ == "__main__":
    main()