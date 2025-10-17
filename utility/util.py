import torch


def find_lora_modules(model):
    lora_layers = []
    for name, m in model.named_modules():
        if hasattr(m, "lora_A") and hasattr(m, "lora_B"):
            lora_layers.append((m, name))
    return lora_layers

def _get_lora_weight_and_grad(lora_module):
    # 1) 如果是 ModuleDict -> 默认取第一个 key（通常是 "default"）
    if isinstance(lora_module, torch.nn.ModuleDict):
        submod = next(iter(lora_module.values()))  # 第一个 Linear 模块
        return submod.weight, submod.weight.grad

    # 2) 如果是 Linear 模块
    if isinstance(lora_module, torch.nn.Linear):
        return lora_module.weight, lora_module.weight.grad

    # 3) 如果是 Parameter
    if isinstance(lora_module, torch.nn.Parameter):
        return lora_module, lora_module.grad

    # 4) 如果直接是 Tensor
    if torch.is_tensor(lora_module):
        return lora_module, getattr(lora_module, "grad", None)

    raise TypeError(f"Unsupported LoRA module type: {type(lora_module)}")