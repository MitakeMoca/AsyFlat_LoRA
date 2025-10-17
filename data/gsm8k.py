import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import re

def extract_final_answer(answer: str) -> str:
    match = re.search(r"####\s*(.*)", answer)
    return match.group(1).strip() if match else answer.strip()

class GSM8k:
    def __init__(self, batch_size, threads, model_name="NousResearch/Meta-Llama-3-8B", data_dir="./datasets"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

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

            # 拼接为完整输入
            full_text = f"Question: {question}\nAnswer: {answer}"

            enc = self.tokenizer(
                full_text,
                max_length=512,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )

            # 提取纯数字答案
            final_answer = extract_final_answer(answer)

            return {
                "input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "labels": enc["input_ids"].squeeze(0),
                "final_answer": final_answer,
                "idx": torch.tensor(idx)
            }
