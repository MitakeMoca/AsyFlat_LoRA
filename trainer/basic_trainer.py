from trainer.trainer import Trainer
import torch

class BaseTrainer(Trainer):
    def train_step(self, batch):
        # batch现在是一个字典
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)
        
        self.optimizer.zero_grad()

        # 前向传播
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels  # 直接传递labels，让模型内部计算loss
        )
        
        loss = outputs.loss
        loss.backward()
        self.optimizer.step()
        return loss.item()
