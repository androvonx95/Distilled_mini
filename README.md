# Distilled Mini - Code Generation Model

A lightweight code generation model built using GPT-2 architecture, specifically designed for code completion and generation tasks.

## Features

- GPT-2 based code generation model with ~112M parameters
- Optimized for resource-constrained environments
- Supports both CPU and GPU training
- DeepSeek data format compatible
- Parameter-efficient fine-tuning capabilities

## Project Structure

```
Distilled_mini/
├── checkpoints/         # Model checkpoints
├── data/               # Training data
├── scripts/            # Data preparation scripts
├── src/                # Source code
│   ├── model.py        # Model architecture
│   ├── tokenizer.py    # Tokenizer configuration
│   ├── model_config.py # Model hyperparameters
│   ├── train.py        # Training script
│   └── evaluate.py     # Evaluation script
└── requirements.txt    # Project dependencies
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Preparation

1. Prepare your training data in JSONL format with prompt-completion pairs:
```jsonl
{
    "prompt": "def add_numbers(a, b):",
    "completion": "\n    return a + b"
}
```

2. Run the data preparation script:
```bash
python scripts/prepare_data.py
```

## Training

Start training with:
```bash
python src/train.py
```

Training configuration:
- Batch Size: 1 (with gradient accumulation)
- Learning Rate: 5e-4
- Epochs: 3
- Checkpoints saved in `checkpoints/` directory

## Model Architecture

- Base Model: GPT-2
- Embedding Size: 1024
- Number of Layers: 16
- Attention Heads: 16
- Total Parameters: ~112M
- Context Window: 2048 tokens
- Vocabulary Size: 50257 tokens

## Tokenizer

- Uses GPT-2's Byte Pair Encoding (BPE) tokenizer
- Maximum sequence length: 1024 tokens
- EOS token used as pad token

## Evaluation

Evaluate the model using:
```bash
python src/evaluate.py
```

## Resource Requirements

- Minimum: 8GB RAM
- Recommended: 16GB RAM
- GPU support available (CUDA)
- CPU fallback supported

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- HuggingFace Transformers
- PyTorch
- PEFT (Parameter-Efficient Fine-Tuning)
- DeepSeek data format
