from utility.util import _get_lora_weight_and_grad, find_lora_modules
import torch
import torch.nn.functional as F

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
        
        # 初始化 EMA 扰动
        self.ema_EB_dict = {}
        for m, name in self.lora_layers:
            if isinstance(m.lora_B, torch.nn.ModuleDict):
                # ModuleDict 结构
                self.ema_EB_dict[name] = {
                    k: torch.zeros_like(v.weight.data) 
                    for k, v in m.lora_B.items()
                }
            else:
                # Linear/Parameter 结构
                self.ema_EB_dict[name] = torch.zeros_like(m.lora_B.data)

    def train_step(self, batch):
        model = self.model
        model.train()

        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        labels = batch["labels"].clone().to(self.device)
        # 确保 pad 被忽略
        labels[labels == getattr(self.tokenizer, "pad_token_id", -100)] = -100

        # === Step 1: 应用 EMA 扰动 ===
        with torch.no_grad():
            for (m, name) in self.lora_layers:
                ema_EB = self.ema_EB_dict[name]
                if isinstance(m.lora_B, torch.nn.ModuleDict):
                    # ModuleDict 结构
                    if not hasattr(m, "_orig_lora_B"):
                        m._orig_lora_B = {k: v.weight.data.clone() for k, v in m.lora_B.items()}
                    for k, v in m.lora_B.items():
                        v.weight.data.add_(ema_EB[k].to(v.weight.device))
                else:
                    # Linear/Parameter 结构
                    if not hasattr(m, "_orig_lora_B"):
                        m._orig_lora_B = m.lora_B.data.clone()
                    m.lora_B.data.add_(ema_EB.to(m.lora_B.device))

        # === Step 2: 在扰动点计算梯度 ===
        self.optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()

        # === Step 3: 计算当前步骤的扰动 EB ===
        rho = self.rho_schedule.step()
        current_EB_dict = {}
        
        for (m, name) in self.lora_layers:
            A, grad_A = _get_lora_weight_and_grad(m.lora_A)
            B, grad_B = _get_lora_weight_and_grad(m.lora_B)

            grad_A = grad_A
            grad_B = grad_B
            s = 0.5  # LoRA scaling factor

            # 估计全参数空间的梯度
            gW = estimate_full_weight_grad(A.detach(), grad_B.detach(), B.detach(), grad_A.detach(), s)
            # 计算全参数空间的扰动
            EW = compute_ew_from_gw(gW, rho)
            # 将扰动转换到低秩空间
            EB = compute_EB_from_EW_and_A(EW, A.detach(), s)
            current_EB_dict[name] = EB

        # === Step 4: 恢复原始参数 ===
        with torch.no_grad():
            for (m, name) in self.lora_layers:
                if hasattr(m, "_orig_lora_B"):
                    if isinstance(m.lora_B, torch.nn.ModuleDict):
                        for k, v in m.lora_B.items():
                            v.weight.data.copy_(m._orig_lora_B[k])
                    else:
                        m.lora_B.data.copy_(m._orig_lora_B)
                    del m._orig_lora_B

        # === Step 5: 使用当前梯度更新参数 ===
        self.optimizer.step()
        self.optimizer.zero_grad()

        # === Step 6: 更新 EMA 扰动 ===
        with torch.no_grad():
            for name, current_EB in current_EB_dict.items():
                ema_EB = self.ema_EB_dict[name]
                if isinstance(ema_EB, dict):
                    # ModuleDict 结构
                    for k in ema_EB.keys():
                        ema_EB[k] = (1 - self.beta) * ema_EB[k] + self.beta * current_EB
                else:
                    # Linear/Parameter 结构
                    self.ema_EB_dict[name] = (1 - self.beta) * ema_EB + self.beta * current_EB

        return loss.item()