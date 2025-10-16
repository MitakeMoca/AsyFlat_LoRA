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
