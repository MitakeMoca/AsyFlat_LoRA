import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple

# 估计 full-weight grad
def estimate_full_weight_grad(A, grad_B, B, grad_A, s):    
    term1 = torch.transpose(    
        torch.linalg.lstsq(A, (1 / s) * grad_B.T).solution, 0, 1    
    ).contiguous()  
        
    term2 = torch.linalg.lstsq(B.T, grad_A).solution.contiguous()  
        
    return 0.5 * (term1 + term2)  
  
  
def compute_ew_from_gw(gW, rho, eps=1e-12):  
    g_flat = gW.reshape(-1)  # 使用 reshape 替代 view  
    norm = torch.norm(g_flat) + eps  
    return (rho * g_flat / norm).view_as(gW)  
  
  
def compute_EB_from_EW_and_A(EW, A, s):    
    return torch.transpose(    
        torch.linalg.lstsq(A.T, (1.0 / s) * EW.T).solution, 0, 1    
    ).contiguous()