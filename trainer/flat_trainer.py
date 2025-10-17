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

    def train_step(self, batch, epoch=0, step=0):
        model = self.model
        model.train()

        inputs, targets, index = [x.to(self.device) for x in batch]

        # === Step 1: baseline forward ===
        self.optimizer.zero_grad()
        outputs = model(inputs).logits
        loss = F.cross_entropy(
            outputs.view(-1, outputs.size(-1)),
            targets.view(-1),
            ignore_index=self.tokenizer.pad_token_id,
            label_smoothing=0.1
        )
        loss.backward(retain_graph=True)

        # === Step 2: compute perturbation EB for each LoRA layer ===
        layer_EBs = []
        for (m, name) in self.lora_layers:
            A, grad_A = _get_lora_weight_and_grad(m.lora_A)
            B, grad_B = _get_lora_weight_and_grad(m.lora_B)

            grad_A = grad_A if grad_A is not None else torch.zeros_like(A)
            grad_B = grad_B if grad_B is not None else torch.zeros_like(B)
            s = getattr(m, "alpha", 1.0)

            gW = estimate_full_weight_grad(A.detach(), grad_B.detach(), B.detach(), grad_A.detach(), s)
            rho = self.rho if self.rho_schedule is None else self.rho_schedule(epoch, step)
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
        outputs2 = model(inputs).logits
        loss2 = F.cross_entropy(
            outputs2.view(-1, outputs2.size(-1)),
            targets.view(-1),
            ignore_index=self.tokenizer.pad_token_id,
            label_smoothing=0.1
        )
        loss2.backward()
        self.optimizer.step()

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

        self.optimizer.zero_grad()
        return loss2.item()