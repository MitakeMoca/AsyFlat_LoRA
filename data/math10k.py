import json
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer


class Math10k:
    def __init__(self, batch_size, threads, model_name="NousResearch/Meta-Llama-3-8B", data_dir="./datasets"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        train_path = f"{data_dir}/math_10k.json"
        train_data = self._load_data(train_path)
        train_set = self._Dataset(train_data, self.tokenizer)

        self.train = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=threads)

    def _load_data(self, path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    class _Dataset(Dataset):
        def __init__(self, data, tokenizer):
            self.data = data
            self.tokenizer = tokenizer

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            example = self.data[idx]

            question_text = example["instruction"].strip()
            if example.get("input", "").strip():
                question_text += "\n" + example["input"].strip()

            answer_text = example["answer"].strip() + "\n" + "Explanation: " + example["output"].strip()

            input_text = f"Question: {question_text}\nAnswer:"

            target_text = " " + answer_text  # 加一个空格以避免紧贴 "Answer:"

            q_enc = self.tokenizer(
                input_text,
                max_length=512,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )

            a_enc = self.tokenizer(
                target_text,
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
