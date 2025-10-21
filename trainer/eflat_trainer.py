from utility.util import _get_lora_weight_and_grad, find_lora_modules
import torch
import torch.nn.functional as F
import time
from collections import defaultdict

from utility.math_util import compute_EB_from_EW_and_A, compute_ew_from_gw, estimate_full_weight_grad


class EFlatLoRATrainer:
    def __init__(self, model, tokenizer, optimizer, device, rho=0.05, rho_schedule=None, beta=0.9):
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.device = device
        self.rho = rho
        self.rho_schedule = rho_schedule
        self.beta = beta
        self.lora_layers = find_lora_modules(model)
        self.time = defaultdict(float)

        # 初始化 EMA 扰动
        self.ema_EB_dict = {}
        for m, name in self.lora_layers:
            if isinstance(m.lora_B, torch.nn.ModuleDict):
                self.ema_EB_dict[name] = {
                    k: torch.zeros_like(v.weight.data)
                    for k, v in m.lora_B.items()
                }
            else:
                self.ema_EB_dict[name] = torch.zeros_like(m.lora_B.data)

    def train_step(self, batch):
        model = self.model
        model.train()

        start_time = time.time()
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        labels = batch["labels"].clone().to(self.device)
        labels[labels == getattr(self.tokenizer, "pad_token_id", -100)] = -100
        self.time["init"] += time.time() - start_time

        # === Step 1: 应用 EMA 扰动 ===
        start_time = time.time()
        with torch.no_grad():
            for (m, name) in self.lora_layers:
                ema_EB = self.ema_EB_dict[name]
                if isinstance(m.lora_B, torch.nn.ModuleDict):
                    if not hasattr(m, "_orig_lora_B"):
                        m._orig_lora_B = {k: v.weight.data.cpu().clone() for k, v in m.lora_B.items()}
                    for k, v in m.lora_B.items():
                        v.weight.data.add_(ema_EB[k].to(v.weight.device))
                else:
                    if not hasattr(m, "_orig_lora_B"):
                        m._orig_lora_B = m.lora_B.data.cpu().clone()
                    m.lora_B.data.add_(ema_EB.to(m.lora_B.device))
        self.time["add_EB"] += time.time() - start_time

        # === Step 2: 在扰动点计算梯度 ===
        start_time = time.time()
        self.optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        self.time["backward"] += time.time() - start_time

        # === Step 3: 计算当前步骤的扰动 EB ===
        start_time = time.time()
        if self.rho_schedule is not None:
            rho = self.rho_schedule.step()  # 外部定义余弦或线性衰减
        else:
            rho = self.rho

        current_EB_dict = {}
        for (m, name) in self.lora_layers:
            A, grad_A = _get_lora_weight_and_grad(m.lora_A)
            B, grad_B = _get_lora_weight_and_grad(m.lora_B)

            grad_A = grad_A
            grad_B = grad_B
            s = 0.5

            # 估计全参数空间的梯度
            gW = estimate_full_weight_grad(A.detach(), grad_B.detach(), B.detach(), grad_A.detach(), s)
            # 计算扰动
            EW = compute_ew_from_gw(gW, rho)
            EB = compute_EB_from_EW_and_A(EW, A.detach(), s)

            if isinstance(m.lora_B, torch.nn.ModuleDict):
                EB_dict = {}
                offset = 0
                for k, v in m.lora_B.items():
                    num = v.weight.numel()
                    EB_dict[k] = EB.view(-1)[offset:offset + num].view_as(v.weight)
                    offset += num
                current_EB_dict[name] = EB_dict
            else:
                current_EB_dict[name] = EB
        self.time["calc_EB"] += time.time() - start_time

        # === Step 4: 恢复原始参数 ===
        start_time = time.time()
        with torch.no_grad():
            for (m, name) in self.lora_layers:
                if hasattr(m, "_orig_lora_B"):
                    if isinstance(m.lora_B, torch.nn.ModuleDict):
                        for k, v in m.lora_B.items():
                            v.weight.data.copy_(m._orig_lora_B[k].to(v.weight.device))
                    else:
                        m.lora_B.data.copy_(m._orig_lora_B.to(m.lora_B.device))
                    del m._orig_lora_B
        self.time["restore"] += time.time() - start_time

        # === Step 5: 优化器更新 ===
        start_time = time.time()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.time["optimize"] += time.time() - start_time

        # === Step 6: EMA 更新 ===
        start_time = time.time()
        with torch.no_grad():
            for name, current_EB in current_EB_dict.items():
                ema_EB = self.ema_EB_dict[name]
                if isinstance(ema_EB, dict):
                    for k in ema_EB.keys():
                        ema_EB[k] = (1 - self.beta) * ema_EB[k] + self.beta * current_EB[k].to(ema_EB[k].device)
                else:
                    self.ema_EB_dict[name] = (1 - self.beta) * ema_EB + self.beta * current_EB.to(ema_EB.device)
        self.time["update_ema"] += time.time() - start_time

        return loss.item()
