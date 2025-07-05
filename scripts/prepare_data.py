import sys
import os

# Add the root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.tokenizer import get_tokenizer
import json
# from src.tokenizer import get_tokenizer

tokenizer = get_tokenizer()

def encode_example(prompt, completion):
    full = prompt + completion
    tokens = tokenizer(full, truncation=True, max_length=2048, return_tensors="pt")
    return {"input_ids": tokens["input_ids"][0], "attention_mask": tokens["attention_mask"][0]}

#   The input_ids part gets us the input tokens 
#   The attention mask is1s and 0s to ignore the padding 
#   Tokenizer output is of the shape [ no_of_sentences , max_sequence_length ]

import json

def main():
    # Read and process the data
    tokenized_data = []
    
    with open("data/distilled_data.jsonl", 'r') as f:
        for line in f:
            try:
                item = json.loads(line.strip())
                if not isinstance(item, dict):
                    continue
                    
                prompt = item.get("prompt", "")
                completion = item.get("completion", "")
                
                if prompt and completion:
                    tokenized_data.append(encode_example(prompt, completion))
            except json.JSONDecodeError:
                print(f"Skipping malformed line: {line.strip()[:50]}...")
            except Exception as e:
                print(f"Error processing line: {e}")
    
    import torch
    torch.save(tokenized_data, "data/tokenized_data.pt")
    print(f"Successfully processed {len(tokenized_data)} valid examples.")

if __name__ == "__main__":
    main()
