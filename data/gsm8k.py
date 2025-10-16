import json
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from datasets import load_dataset


class GSM8k:
    def __init__(self, batch_size, threads, model_name="gpt2", data_dir="./datasets"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        test_path = f"{data_dir}/gsm_8k.json"
        test_data = self._load_data(test_path)

        test_set = self._Dataset(test_data, self.tokenizer)

        self.test = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=threads)

    def _load_data(self, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data

    class _Dataset(Dataset):
        def __init__(self, data, tokenizer):
            self.data = data
            self.tokenizer = tokenizer

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            example = self.data[idx]
            question = example["question"]
            answer = example["answer"]

            q_enc = self.tokenizer(
                question,
                max_length=512,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )

            a_enc = self.tokenizer(
                answer,
                max_length=512,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )

            return (
                q_enc["input_ids"].squeeze(0),
                a_enc["input_ids"].squeeze(0),
                torch.tensor(idx, dtype=torch.long)
            )
