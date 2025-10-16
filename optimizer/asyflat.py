import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class AsyFlat_LoRA(torch.optim.Optimizer):
    def __init__(self, params: list[torch.nn.parameter.Parameter], base_optimizer: torch.optim.Optimizer, rho, rho_scheduler, adaptive, storage_size, alpha, beta, **kwargs):
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(AsyFlat_LoRA, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer
        self.rho_scheduler = rho_scheduler
        self.param_groups = self.base_optimizer.param_groups
        self.rho = rho
        self.alpha = alpha
        self.beta = beta
        self.impt = torch.ones((storage_size, 3)).to(device)
    
    # 用来对权重梯度进行线性归一化
    def norm_to_prob(self, tensor, scale_min=0.1, scale_max=0.9):
        # 缩放到 0.1-0.9 范围
        scaled_tensor = (tensor - torch.min(tensor)) / (torch.max(tensor) - torch.min(tensor)) * (
                scale_max - scale_min) + scale_min

        # 计算概率值，调整为和为1
        sum_scaled_tensor = torch.sum(scaled_tensor)
        probabilities = scaled_tensor / sum_scaled_tensor

        return probabilities
    
    @torch.no_grad()
    def update_rho(self):
        self.rho = self.rho_scheduler.step()
        return self.rho
    
    # 采样数据
    @torch.no_grad()
    def sample_index(self, args, epoch, index, fmax_):
        data_len = len(index)
        sample_size = int(self.alpha * data_len)

        if epoch == 0:
            tf = torch.arange(data_len)
        else:
            prob_distribution = self.norm_to_prob(self.impt[index, 2], scale_min=args["fmin"], scale_max=fmax_)
            tf = torch.multinomial(prob_distribution, sample_size, replacement=False)
        return tf
    
    # 更新 impt 数组
    @torch.no_grad()
    def impt_roc(self, epoch, index, tf, roc):
        selected_index = index[tf]
        if epoch == 0:
            self.impt[selected_index, 0] = roc
            self.impt[selected_index, 2] = roc
        else:
            self.impt[selected_index, 0] = self.impt[selected_index, 0] + roc
            self.impt[selected_index, 1] = self.impt[selected_index, 1] + 1
            self.impt[selected_index, 2] = self.impt[selected_index, 0] / self.impt[selected_index, 1]

    # 第一步增加扰动
    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        scale = self.rho / (grad_norm + 1e-12)
        for group in self.param_groups:
            # p 就是扰动后的目标了
            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = p.grad * scale.to(p)
                p.add_(e_w)
        if zero_grad: self.zero_grad()
    
    @torch.no_grad()
    def second_step_without_norm(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]

        self.base_optimizer.step()
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step_with_norm(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]
                p.grad = p.grad * self.norm1

        self.base_optimizer.step()
        if zero_grad: self.zero_grad()

    # 算两个向量的余弦相似度
    def cul_cos(self, g1, g2):
        inner_prod = 0.0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                inner_prod += torch.sum(
                    self.state[p][g1] * self.state[p][g2]
                )

        # get norm
        g1_grad_norm = self._grad_norm_by(by=g1)
        g2_grad_norm = self._grad_norm_by(by=g2)
        cos_g1_g2 = inner_prod / (g1_grad_norm * g2_grad_norm + 0.00000000000001)

        return cos_g1_g2

    def sgd_step(self, zero_grad=False):
        self.base_optimizer.step()
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    @torch.no_grad()
    def _grad_norm_by(self, by=None, weight_adaptive=False):
        # shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        if not by:
            norm = torch.norm(
                torch.stack([
                    ((torch.abs(p.data) if weight_adaptive else 1.0) * p.grad).norm(p=2)
                    for group in self.param_groups for p in group["params"]
                    if p.grad is not None
                ]),
                p=2
            )
        else:
            norm = torch.norm(
                torch.stack([
                    ((torch.abs(p.data) if weight_adaptive else 1.0) * self.state[p][by]).norm(p=2)
                    for group in self.param_groups for p in group["params"]
                    if p.grad is not None
                ]),
                p=2
            )
        return norm

    def _grad_norm_by_layer(self, by=None, weight_adaptive=False):
        if not by:
            norm = torch.norm(
                torch.stack([
                    ((torch.abs(p.data) if weight_adaptive else 1.0) * p.grad).norm(p=2)
                    for group in self.param_groups for p in group["params"]
                    if p.grad is not None
                ]),
                p=2
            )
        else:
            norm = torch.norm(
                torch.stack([
                    ((torch.abs(p.data) if weight_adaptive else 1.0) * self.state[p][by]).norm(p=2)
                    for group in self.param_groups for p in group["params"]
                    if p.grad is not None
                ]),
                p=2
            )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups