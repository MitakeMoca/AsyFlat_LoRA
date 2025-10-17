import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple

# 估计 full-weight grad
def estimate_full_weight_grad(A, grad_B, B, grad_A, s):
    A_T_pinv = torch.pinverse(A.T)  # (r,m) -> pinv -> (m,r)
    B_T_pinv = torch.pinverse(B.T)  # (r,n) -> pinv -> (n,r)
    term1 = (1 / s) * (grad_B @ A_T_pinv)
    term2 = (1 / s) * (B_T_pinv @ grad_A)
    return 0.5 * (term1 + term2)  # (n,m)


def compute_ew_from_gw(gW, rho, eps=1e-12):
    g_flat = gW.view(-1)
    norm = torch.norm(g_flat) + eps
    return (rho * g_flat / norm).view_as(gW)


def compute_EB_from_EW_and_A(EW, A, s):
    A_pinv = torch.pinverse(A)  # (r,m) -> (m,r)
    return (1.0 / s) * (EW @ A_pinv)  # (n,m)@(m,r) -> (n,r)