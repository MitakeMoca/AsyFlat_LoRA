from utility.util import _get_lora_weight_and_grad, find_lora_modules
import torch
import torch.nn.functional as F
import time
from collections import defaultdict

from utility.math_util import compute_EB_from_EW_and_A, compute_ew_from_gw, estimate_full_weight_grad

class AsyFlatTrainer:
    def __init__(self, model, tokenizer, optimizer, device, sample_size, fmin, fmax, rho=0.05, rho_schedule=None, alpha=0.5, beta=0.9):
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.device = device
        self.rho = rho
        self.rho_schedule = rho_schedule
        self.alpha = alpha
        self.beta = beta
        self.lora_layers = find_lora_modules(model)
        self.fmin = fmin
        self.fmax = fmax

        # timing dict
        self.time = defaultdict(float)

        # 初始化 EMA 扰动（结构与 m.lora_B 对应）
        self.ema_EB_dict = {}
        for m, name in self.lora_layers:
            if isinstance(m.lora_B, torch.nn.ModuleDict):
                self.ema_EB_dict[name] = {k: torch.zeros_like(v.weight.data) for k, v in m.lora_B.items()}
            else:
                self.ema_EB_dict[name] = torch.zeros_like(m.lora_B.data)

        # 记录 sample 在扰动点的损失
        self.sample_loss = torch.zeros(sample_size, dtype=torch.float32, device=device)
        self.sample_cnt = torch.zeros(sample_size, dtype=torch.int32, device=device)
        self.sample_avg = torch.zeros(sample_size, dtype=torch.float32, device=device)

    def norm_to_prob(self, tensor, epoch):
        fmin = self.fmin
        fmax = self.fmax
        fmax_ = fmax - ((3 - epoch) / 3) * (fmax - fmin) + 1e-9
        # 缩放到 fmin - fmax_
        denom = (torch.max(tensor) - torch.min(tensor)).clamp(min=1e-9)
        scaled_tensor = (tensor - torch.min(tensor)) / denom * (fmax_ - fmin) + fmin

        # normalize to prob
        sum_scaled = torch.sum(scaled_tensor).clamp(min=1e-9)
        probabilities = scaled_tensor / sum_scaled
        return probabilities

    # 采样数据，返回的是相对索引
    @torch.no_grad()
    def sample_index(self, epoch, index):
        data_len = len(index)
        sample_size = max(1, int(self.alpha * data_len))

        prob_distribution = self.norm_to_prob(self.sample_avg[index], epoch)
        tf = torch.multinomial(prob_distribution, sample_size, replacement=False)
        return tf

    def train_step(self, batch, epoch):
        model = self.model
        model.train()

        start_time = time.time()
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        labels = batch["labels"].clone().to(self.device)
        labels[labels == getattr(self.tokenizer, "pad_token_id", -100)] = -100
        index = batch["idx"].clone().to(self.device)
        self.time["init"] += time.time() - start_time

        # 说明已经进行过一次存取了，所以这次利用梯度信息抽样
        start_time = time.time()
        if epoch != 1:
            tf = self.sample_index(epoch, index)
            input_ids = input_ids[tf]
            if attention_mask is not None:
                attention_mask = attention_mask[tf]
            labels = labels[tf]
            index = index[tf]
        self.time["sample"] += time.time() - start_time

        # === Step 1: 应用 EMA 扰动（备份到 CPU，添加到 GPU） ===
        start_time = time.time()
        with torch.no_grad():
            for (m, name) in self.lora_layers:
                ema_EB = self.ema_EB_dict[name]
                if isinstance(m.lora_B, torch.nn.ModuleDict):
                    # 备份到 CPU（仅第一次）
                    if not hasattr(m, "_orig_lora_B"):
                        m._orig_lora_B = {k: v.weight.data.cpu().clone() for k, v in m.lora_B.items()}
                    # add per-key EB
                    for k, v in m.lora_B.items():
                        v.weight.data.add_(ema_EB[k].to(v.weight.device))
                else:
                    if not hasattr(m, "_orig_lora_B"):
                        m._orig_lora_B = m.lora_B.data.cpu().clone()
                    m.lora_B.data.add_(ema_EB.to(m.lora_B.device))
        self.time["add_EB"] += time.time() - start_time

        # === Step 2: 在扰动点计算抽样样本梯度（按样本 loss） ===
        start_time = time.time()
        self.optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        # 计算 per-sample loss more robustly (vectorized)
        logits = outputs.logits  # [B, T, V]
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()  # ignore_index already -100
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
        per_pos_loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        ).view(shift_labels.shape)  # [B, T-1]

        # mean over valid tokens per sample
        valid_mask = (shift_labels != -100).float()
        valid_counts = valid_mask.sum(dim=1).clamp(min=1.0)
        per_sample_loss = (per_pos_loss * valid_mask).sum(dim=1) / valid_counts  # [B]
        loss = per_sample_loss.mean()
        loss.backward()
        self.time["backward"] += time.time() - start_time

        # === Step 3: 计算当前步骤的扰动 EB （按 module） ===
        start_time = time.time()
        rho = self.rho_schedule.step() if self.rho_schedule is not None else self.rho
        current_EB_dict = {}

        for (m, name) in self.lora_layers:
            A, grad_A = _get_lora_weight_and_grad(m.lora_A)
            B, grad_B = _get_lora_weight_and_grad(m.lora_B)

            # fallback if grad is None
            grad_A = grad_A if grad_A is not None else torch.zeros_like(A)
            grad_B = grad_B if grad_B is not None else torch.zeros_like(B)

            s = 0.5

            # estimate full-weight grad (gW)
            gW = estimate_full_weight_grad(A.detach(), grad_B.detach(), B.detach(), grad_A.detach(), s)
            # compute EW and EB
            EW = compute_ew_from_gw(gW, rho)
            EB = compute_EB_from_EW_and_A(EW, A.detach(), s)  # EB shape matches B shape (n, r)

            if isinstance(m.lora_B, torch.nn.ModuleDict):
                # split EB into per-key tensors consistent with m.lora_B item order
                EB_flat = EB.view(-1)
                offset = 0
                EB_map = {}
                for k, v in m.lora_B.items():
                    num = v.weight.numel()
                    seg = EB_flat[offset: offset + num].view_as(v.weight)
                    EB_map[k] = seg.detach().to(self.device)
                    offset += num
                current_EB_dict[name] = EB_map
            else:
                current_EB_dict[name] = EB.detach().to(self.device)
        self.time["calc_EB"] += time.time() - start_time

        # === Step 4: 恢复原始参数（从 CPU 备份拷回） ===
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

        # === Step 5: 使用当前梯度更新参数 ===
        start_time = time.time()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.time["optimize"] += time.time() - start_time

        # === Step 6: 更新 EMA 扰动（保持与 current_EB 结构一致） ===
        start_time = time.time()
        with torch.no_grad():
            for name, current_EB in current_EB_dict.items():
                ema_EB = self.ema_EB_dict[name]
                if isinstance(ema_EB, dict):
                    # per-key EMA
                    for k in ema_EB.keys():
                        ema_EB[k] = (1 - self.beta) * ema_EB[k] + self.beta * current_EB[k].to(ema_EB[k].device)
                else:
                    self.ema_EB_dict[name] = (1 - self.beta) * ema_EB + self.beta * current_EB.to(ema_EB.device)
        self.time["update_ema"] += time.time() - start_time

        # === Step 7: 更新 sample loss 平均值 ===
        start_time = time.time()
        with torch.no_grad():
            B = index.size(0)
            for i in range(B):
                ii = index[i].item()
                self.sample_loss[ii] += per_sample_loss[i].detach().to(self.sample_loss.device)
                self.sample_cnt[ii] += 1
                self.sample_avg[ii] = self.sample_loss[ii] / self.sample_cnt[ii].clamp(min=1)
        self.time["update_loss"] += time.time() - start_time

        return loss.item()
