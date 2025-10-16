import torch
import torch.nn.functional as F

def smooth_crossentropy(pred, gold, smoothing=0.1, ignore_index=-100):
    """
    Args:
        pred: [B, T, V]
        gold: [B, T]
    Returns:
        loss_per_sample: [B]
    """
    vocab_size = pred.size(-1)
    loss = F.cross_entropy(
        pred.view(-1, vocab_size),
        gold.view(-1),
        label_smoothing=smoothing,
        reduction='none',
        ignore_index=ignore_index
    ).view(gold.size())

    mask = (gold != ignore_index).float()

    loss_per_sample = (loss * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

    return loss_per_sample  # [B]

import re

def extract_final_answer(text: str) -> str:
    """
    提取 '####' 后的最终答案字符串。
    如果不存在 '####'，则返回空字符串。
    会自动去除前后空格、换行、多余的符号。
    """
    if not text:
        return ""
    
    match = re.search(r"####\s*(.+)", text)
    if match:
        ans = match.group(1).strip()
        # 去除可能出现在结尾的标点
        ans = ans.strip(".，。；;")
        return ans
    else:
        return ""
