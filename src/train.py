import sys
import os

# Add the root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from src.model import get_model
from src.tokenizer import get_tokenizer
import torch

def load_data():
    return torch.load("data/tokenized_data.pt")

class CodeDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        item = self.data[idx]
        return {"input_ids": item["input_ids"], "attention_mask": item["attention_mask"], "labels": item["input_ids"]}

    def __len__(self):
        return len(self.data)

def main():
    model = get_model()
    tokenizer = get_tokenizer()
    data = load_data()
    dataset = CodeDataset(data)

    args = TrainingArguments(
        output_dir="./checkpoints",
        num_train_epochs=30,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=16,
        learning_rate=3e-5,
        save_steps=50,
        logging_steps=10,
        use_cpu=not torch.cuda.is_available(),
        report_to="none",
        save_total_limit=5,
        warmup_steps=100,
        weight_decay=0.01,
        fp16=True if torch.cuda.is_available() else False,
        gradient_checkpointing=True,
        optim="adamw_torch",
        dataloader_num_workers=0 if torch.cuda.is_available() else 2,
        dataloader_pin_memory=True if torch.cuda.is_available() else False
    )

    # Enable gradient checkpointing on the model
    model.gradient_checkpointing_enable()
    
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )

    trainer.train()

if __name__ == "__main__":
    main()
