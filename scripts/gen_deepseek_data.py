import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import os
import random
import argparse
from pathlib import Path
from tqdm import tqdm

# === SETTINGS ===
MODEL_NAME = "deepseek-ai/deepseek-coder-1.3b-base"
OUTPUT_FILE = Path("data/deepseek_distilled_50k.jsonl")
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

# === FUNCTION COMPONENTS ===
FUNCTION_NAMES = [
    "add", "subtract", "multiply", "divide", "power", "modulo",
    "is_prime", "factorial", "fibonacci", "gcd", "lcm", "sum_list",
    "average", "median", "mode", "reverse_string", "is_palindrome",
    "flatten", "binary_search", "merge_sort", "quick_sort", "bubble_sort",
    "is_even", "is_odd", "validate_email", "generate_password",
    "read_file", "write_file", "parse_json", "to_camel_case"
]

ARG_PATTERNS = [
    "(a, b)", "(n)", "(arr)", "(lst)", "(x)", "(text)", "(data)", "(s)",
    "(filename)", "(content)", "(email)", "(input_str)", "(values)", "(json_obj)"
]

DOCSTRINGS = [
    "Returns the result of the operation.",
    "Checks whether the input is valid.",
    "Performs a recursive calculation.",
    "Sorts the input list using merge sort.",
    "Searches for a value in a sorted array.",
    "Parses the given text and extracts information.",
    "Reads content from a file.",
    "Writes data to the specified file.",
    "Converts a string to camel case format.",
    "Generates a random password with given constraints."
]

COMMENTS = [
    "# Base case for recursion",
    "# Iterate through the list",
    "# Check if the number is prime",
    "# Use binary search logic",
    "# Convert string to lowercase",
    "# Open the file in read mode",
    "# Handle division by zero",
    "# Calculate the mean value",
    "# Filter out invalid entries",
    "# Return the final result"
]

# === MODEL LOADING ===
print("⏳ Loading DeepSeek model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,
    device_map="auto"
)
model.eval()
print("✅ Model loaded!")

# === GENERATOR ===
def generate_prompt():
    name = random.choice(FUNCTION_NAMES)
    args = random.choice(ARG_PATTERNS)
    use_doc = random.random() < 0.6
    use_comment = random.random() < 0.3

    prompt = f"def {name}{args}:\n"

    if use_doc:
        docstring = random.choice(DOCSTRINGS)
        prompt += f'    """{docstring}"""\n'

    if use_comment:
        comment = random.choice(COMMENTS)
        prompt += f"    {comment}\n"

    return prompt

def generate_completion(prompt, max_new_tokens=128, temperature=0.4):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )
    generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return generated.strip()

def main(total_samples: int = 50000):
    existing = set()
    if OUTPUT_FILE.exists():
        print("🔁 Resuming from existing file...")
        with open(OUTPUT_FILE, "r") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    existing.add(data["prompt"].strip())
                except: pass

    with open(OUTPUT_FILE, "a") as f_out:
        pbar = tqdm(total=total_samples)
        pbar.update(len(existing))

        while len(existing) < total_samples:
            prompt = generate_prompt()
            if prompt in existing:
                continue
            completion = generate_completion(prompt)
            if not completion or len(completion.splitlines()) < 2:
                continue
            entry = {"prompt": prompt.strip(), "completion": completion.strip()}
            f_out.write(json.dumps(entry) + "\n")
            f_out.flush()
            existing.add(prompt)
            pbar.update(1)

        pbar.close()
    print(f"✅ Done! Total saved: {len(existing)} examples")

# === CLI ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=50000, help="Total number of examples to generate")
    args = parser.parse_args()

    main(args.samples)
