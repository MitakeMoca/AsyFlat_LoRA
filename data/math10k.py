import json
import torch
from torch.utils.data import DataLoader, RandomSampler
from transformers import AutoTokenizer

class Math10k:
    def __init__(self, batch_size, threads, model_name="gpt2", data_dir="/datasets"):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        def load_data(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data

        def encode_fn(examples):
            q_enc = tokenizer(
                examples["question"],
                max_length=512,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )
            a_enc = tokenizer(
                examples["answer"],
                max_length=512,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )
            return {
                "input_ids": q_enc["input_ids"].squeeze(0),
                "attention_mask": q_enc["attention_mask"].squeeze(0),
                "labels": a_enc["input_ids"].squeeze(0)
            }

        class _Dataset(torch.utils.data.Dataset):
            def __init__(self, data):
                self.data = data
            def __len__(self): return len(self.data)
            def __getitem__(self, idx): return encode_fn(self.data[idx])

        train_data = load_data(f"{data_dir}/math_10k.json")

        self.train = DataLoader(_Dataset(train_data), batch_size=batch_size, 
                                shuffle=True, num_workers=threads)
