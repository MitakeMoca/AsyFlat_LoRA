from utility.util import _get_lora_weight_and_grad, find_lora_modules
import torch
import torch.nn.functional as F

from utility.math_util import compute_EB_from_EW_and_A, compute_ew_from_gw, estimate_full_weight_grad

class FlatLoRATrainer:
    def __init__(self, model, tokenizer, optimizer, device, rho=0.05, rho_schedule=None):
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.device = device
        self.rho = rho
        self.rho_schedule = rho_schedule
        self.lora_layers = find_lora_modules(model)

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

        # === Step 1: baseline forward ===
        self.optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()

        # === Step 2: compute perturbation EB for each LoRA layer ===
        layer_EBs = []
        rho = self.rho_schedule.step()
        for (m, name) in self.lora_layers:
            A, grad_A = _get_lora_weight_and_grad(m.lora_A)
            B, grad_B = _get_lora_weight_and_grad(m.lora_B)

            grad_A = grad_A
            grad_B = grad_B
            s = 0.5

            gW = estimate_full_weight_grad(A.detach(), grad_B.detach(), B.detach(), grad_A.detach(), s)
            EW = compute_ew_from_gw(gW, rho)
            EB = compute_EB_from_EW_and_A(EW, A.detach(), s)
            layer_EBs.append((m, EB))

        # === Step 3: add EB perturbation to B ===
        with torch.no_grad():
            for (m, EB) in layer_EBs:
                # ModuleDict 结构
                if isinstance(m.lora_B, torch.nn.ModuleDict):
                    if not hasattr(m, "_orig_lora_B"):
                        m._orig_lora_B = {k: v.weight.data.clone() for k, v in m.lora_B.items()}
                    for k, v in m.lora_B.items():
                        v.weight.data.add_(EB.to(v.weight.device))
                # Linear/Parameter 结构
                else:
                    if not hasattr(m, "_orig_lora_B"):
                        m._orig_lora_B = m.lora_B.data.clone()
                    m.lora_B.data.add_(EB.to(m.lora_B.device))

        # === Step 4: second forward/backward (perturbed point) ===
        self.optimizer.zero_grad()
        outputs2 = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss2 = outputs2.loss
        loss2.backward()
        
        # === Step 5: restore original B ===
        with torch.no_grad():
            for (m, _) in layer_EBs:
                if hasattr(m, "_orig_lora_B"):
                    if isinstance(m.lora_B, torch.nn.ModuleDict):
                        for k, v in m.lora_B.items():
                            v.weight.data.copy_(m._orig_lora_B[k])
                    else:
                        m.lora_B.data.copy_(m._orig_lora_B)
                    del m._orig_lora_B

        self.optimizer.step()

        self.optimizer.zero_grad()
        return loss2.item()