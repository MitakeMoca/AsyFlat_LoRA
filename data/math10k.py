import json
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer


class Math10k:
    def __init__(self, batch_size, threads, model_name="gpt2", data_dir="./datasets"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        train_path = f"{data_dir}/math_10k.json"
        train_data = self._load_data(train_path)

        train_set = self._Dataset(train_data, self.tokenizer)

        self.train = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=threads)


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
            question_text = example["instruction"]
            if example.get("input", "").strip():
                question_text += "\n" + example["input"]

            q_enc = self.tokenizer(
                question_text,
                max_length=512,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )

            a_enc = self.tokenizer(
                example["output"],
                max_length=512,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )

            return (
                q_enc["input_ids"].squeeze(0),  # inputs
                a_enc["input_ids"].squeeze(0),  # targets
                torch.tensor(idx, dtype=torch.long)  # index
            )
