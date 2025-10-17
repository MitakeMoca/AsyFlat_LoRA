# 训练器的统一接口

class Trainer:
    def __init__(self, model, tokenizer, optimizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.device = device

    def train_step(self, batch):
        raise NotImplementedError
