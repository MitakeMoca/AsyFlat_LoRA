from utility.util import _get_lora_weight_and_grad, find_lora_modules
import torch
import torch.nn.functional as F

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

        # 记录 sample 在扰动点的损失
        self.sample_loss = torch.zeros(sample_size, dtype=torch.float32, device=device)
        self.sample_cnt = torch.zeros(sample_size, dtype=torch.int32, device=device)
        self.sample_avg = torch.zeros(sample_size, dtype=torch.float32, device=device)

    def norm_to_prob(self, tensor, epoch):
        fmin = self.fmin
        fmax = self.fmax
        fmax_ = fmax - ((3 - epoch) / 3) * (fmax - fmin) + 0.000000001
        # 缩放到 0.1-0.9 范围
        scaled_tensor = (tensor - torch.min(tensor)) / (torch.max(tensor) - torch.min(tensor) + 0.000000001) * (
                fmax_ - fmin) + fmin

        # 计算概率值，调整为和为1
        sum_scaled_tensor = torch.sum(scaled_tensor)
        probabilities = scaled_tensor / sum_scaled_tensor

        return probabilities
    
    # 采样数据，返回的是相对索引
    @torch.no_grad()
    def sample_index(self, epoch, index):
        data_len = len(index)
        sample_size = int(self.alpha * data_len)

        prob_distribution = self.norm_to_prob(self.sample_avg[index], epoch)
        tf = torch.multinomial(prob_distribution, sample_size, replacement=False)
        return tf

    def train_step(self, batch, epoch):
        model = self.model
        model.train()

        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        labels = batch["labels"].clone().to(self.device)
        # 确保 pad 被忽略
        labels[labels == getattr(self.tokenizer, "pad_token_id", -100)] = -100
        index = batch["idx"].clone().to(self.device)

        # 说明已经进行过一次存取了，所以这次利用梯度信息抽样
        if epoch != 1:
            tf = self.sample_index(epoch, index)
            input_ids = input_ids[tf]
            attention_mask = attention_mask[tf]
            labels = labels[tf]
            index = index[tf]

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

        # === Step 2: 在扰动点计算抽样样本梯度 ===
        self.optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        # 方法2：直接使用模型返回的损失，但分解为每个样本
        logits = outputs.logits
        # 移位处理
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # 计算每个位置的损失
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
        per_position_loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), 
            shift_labels.view(-1)
        ).view(shift_labels.shape)
        # 手动求每个样本的 loss
        batch_size = labels.size(0)
        per_sample_loss = []
        for i in range(batch_size):
            valid_indices = (shift_labels[i] != -100)
            if valid_indices.sum() > 0:
                sample_loss = per_position_loss[i][valid_indices].mean()
            else:
                sample_loss = torch.tensor(0.0, device=per_position_loss.device)
            per_sample_loss.append(sample_loss)
        per_sample_loss = torch.stack(per_sample_loss)
        loss = per_sample_loss.mean()
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

        # === Step 7: 更新梯度范数估计值 ===
        with torch.no_grad():
            nn = len(index)
            for i in range(nn):
                ii = index[i]
                self.sample_loss[ii] += per_sample_loss[i]
                self.sample_cnt[ii] += 1
                self.sample_avg[ii] = self.sample_loss[ii] / self.sample_cnt[ii]

        return loss.item()


        
