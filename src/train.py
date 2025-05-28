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
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=5e-4,
        save_steps=100,
        logging_steps=10,
        no_cuda=not torch.cuda.is_available(),   #no_cuda=True,
        evaluation_strategy="no",
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )

    trainer.train()

if __name__ == "__main__":
    main()
