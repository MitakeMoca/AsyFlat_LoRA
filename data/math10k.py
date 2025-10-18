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
        def __init__(self, data, tokenizer, max_length=512):
            self.data = data
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            example = self.data[idx]
            question_text = example["instruction"].strip()
            if example.get("input", "").strip():
                question_text += "\n" + example["input"].strip()
            
            answer_text = example["answer"].strip() + "\n" + "Explanation: " + example["output"].strip()
            
            # 构建完整序列
            full_text = f"Question: {question_text}\nAnswer: {answer_text}"
            
            enc = self.tokenizer(
                full_text,
                max_length=512,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )
            
            input_ids = enc["input_ids"].squeeze(0)
            attention_mask = enc["attention_mask"].squeeze(0)
            
            # 创建labels：只计算答案部分的loss
            labels = input_ids.clone()
            
            # 找到"Answer:"的位置，之前的部分设为-100
            text_before_answer = f"Question: {question_text}\nAnswer:"
            answer_start = len(self.tokenizer.encode(text_before_answer, add_special_tokens=False))
            labels[:answer_start] = -100
            
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "idx": torch.tensor(idx, dtype=torch.long)
            }

