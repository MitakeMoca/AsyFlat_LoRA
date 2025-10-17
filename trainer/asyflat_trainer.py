from trainer.trainer import Trainer
from utility.bypass_bn import disable_running_stats, enable_running_stats
from utility.loss import smooth_crossentropy
import torch

class AsyFlatTrainer(Trainer):
    def __init__(self, model, tokenizer, optimizer, asyflat_optimizer, device):
        super().__init__(model, tokenizer, optimizer, device)
        self.asyflat_optimizer = asyflat_optimizer

    def train_step(self, batch, epoch, fmax_):
        inputs, targets, index = [x.to(self.device) for x in batch]

        tf = self.asyflat_optimizer.sample_index(None, epoch, index, fmax_)

        enable_running_stats(self.model)
        loss_bef = smooth_crossentropy(self.model(inputs[tf]).logits, targets[tf])
        loss_bef.mean().backward()
        self.asyflat_optimizer.first_step(zero_grad=True)

        disable_running_stats(self.model)
        loss_aft = smooth_crossentropy(self.model(inputs[tf]).logits, targets[tf])
        loss_aft.mean().backward()
        self.asyflat_optimizer.second_step_without_norm(zero_grad=True)

        roc = torch.abs(loss_aft - loss_bef)
        self.asyflat_optimizer.impt_roc(epoch, index, tf, roc)
        return loss_aft.mean().item()
