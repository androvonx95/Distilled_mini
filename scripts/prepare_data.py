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

def main():
    with open("data/distilled_data.jsonl") as f:
        raw_data = [json.loads(line) for line in f]

    tokenized_data = [encode_example(d["prompt"], d["completion"]) for d in raw_data]

    import torch
    torch.save(tokenized_data, "data/tokenized_data.pt")
    print(f"Saved {len(tokenized_data)} examples.")

if __name__ == "__main__":
    main()
