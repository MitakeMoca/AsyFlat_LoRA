from trainer.trainer import Trainer
import torch

class BaseTrainer(Trainer):
    def train_step(self, batch):
        inputs, targets, index = [x.to(self.device) for x in batch]
        self.optimizer.zero_grad()

        outputs = self.model(inputs).logits
        loss = torch.nn.functional.cross_entropy(
            outputs.view(-1, outputs.size(-1)),
            targets.view(-1),
            ignore_index=self.tokenizer.pad_token_id,
            label_smoothing=0.1,
        )
        loss.backward()
        self.optimizer.step()
        return loss.item()
