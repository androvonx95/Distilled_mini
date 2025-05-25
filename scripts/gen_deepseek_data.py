import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import os
from pathlib import Path

# Adjust this to your local DeepSeek Coder path or HuggingFace name
MODEL_NAME = "deepseek-ai/deepseek-coder-1.3b-base"

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,
    device_map="auto"
)
model.eval()
print("Model loaded.")

# Prompts to generate code completions for
function_prompts = [
    "def fibonacci(n):",
    "def is_prime(x):",
    "def factorial(n):",
    "def quicksort(arr):",
    "def is_palindrome(s):",
    "def reverse_string(text):",
    "def gcd(a, b):",
    "def flatten(nested_list):",
    "def binary_search(arr, target):",
    "def count_words(text):"
]

OUTPUT_FILE = Path("data/deepseek_distilled.jsonl")
OUTPUT_FILE.parent.mkdir(exist_ok=True, parents=True)

def generate_completion(prompt, max_new_tokens=100, temperature=0.4):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.95,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return generated

def main():
    with open(OUTPUT_FILE, "w") as f:
        for idx, prompt in enumerate(function_prompts):
            completion = generate_completion(prompt)
            entry = {"prompt": prompt, "completion": completion}
            f.write(json.dumps(entry) + "\n")
            print(f"[{idx+1}] {prompt.strip()} ✅")

if __name__ == "__main__":
    main()
