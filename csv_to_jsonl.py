import pandas as pd
import json

def csv_to_jsonl(input_csv_path, output_jsonl_path):
    """
    Convert tab-separated CSV to JSONL format
    """
    # Read the CSV file with tab separator
    df = pd.read_csv(input_csv_path, sep='\t', header=None, names=['id', 'sample_id', 'question_text'])
    
    # Convert to JSONL
    with open(output_jsonl_path, 'w', encoding='utf-8') as f:
        for _, row in df.iterrows():
            record = {
                'id': row['id'],
                'sample_id': row['sample_id'],
                'question': row['question_text']
            }
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    print(f"Converted {len(df)} records to {output_jsonl_path}")

if __name__ == "__main__":
    input_path = "/data/zzy/CoT-compression/data/input.csv"
    output_path = "/data/zzy/CoT-compression/data/input.jsonl"
    csv_to_jsonl(input_path, output_path)