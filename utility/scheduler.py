import math
import numpy as np

class ProportionScheduler:
    def __init__(self, pytorch_lr_scheduler, max_lr, min_lr, max_value, min_value):
        self.t = 0    
        self.pytorch_lr_scheduler = pytorch_lr_scheduler
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.max_value = max_value
        self.min_value = min_value
        
        self.step() # take 1 step during initialization to get self._last_lr
    
    def lr(self):
        return self._last_lr[0]
                
    def step(self):
        self.t += 1
        if hasattr(self.pytorch_lr_scheduler, "_last_lr"):
            lr = self.pytorch_lr_scheduler._last_lr[0]
        else:
            lr = self.pytorch_lr_scheduler.optimizer.param_groups[0]['lr']
            
        if self.max_lr > self.min_lr:
            value = self.min_value + (self.max_value - self.min_value) * (lr - self.min_lr) / (self.max_lr - self.min_lr)
        else:
            value = self.max_value
        
        self._last_lr = [value]
        return value
        
class SchedulerBase:
    def __init__(self, T_max, max_value, min_value=0.0, init_value=0.0, warmup_steps=0, optimizer=None):
        super(SchedulerBase, self).__init__()
        self.t = 0
        self.min_value = min_value
        self.max_value = max_value
        self.init_value = init_value
        self.warmup_steps = warmup_steps
        self.total_steps = T_max
        
        self._last_lr = [init_value]
                
        self.optimizer = optimizer

    def step(self):
        if self.t < self.warmup_steps:
            value = self.init_value + (self.max_value - self.init_value) * self.t / self.warmup_steps
        elif self.t == self.warmup_steps:
            value = self.max_value
        else:
            value = self.step_func()
        self.t += 1

        if self.optimizer is not None:
            for param_group in self.optimizer.param_groups:
                # if self.t <
                param_group['lr'] = value
                
        self._last_lr = [value]
        return value

    def step_func(self):
        pass
    
    def lr(self):
        return self._last_lr[0]

class CosineScheduler(SchedulerBase):
    def step_func(self):
        phase = (self.t-self.warmup_steps) / (self.total_steps-self.warmup_steps) * math.pi
        value = self.min_value + (self.max_value-self.min_value) * (np.cos(phase) + 1.) / 2.0
        return value
    
import math

class CosineRhoScheduler:
    def __init__(self, max_value, min_value, total_steps):
        """
        一个余弦衰减的 ρ 调度器。

        Args:
            max_value (float): 初始最大 ρ
            min_value (float): 最小 ρ（衰减下限）
            total_steps (int): 总步数（epoch * steps_per_epoch）
        """
        self.max_value = max_value
        self.min_value = min_value
        self.total_steps = total_steps
        self.current_step = 0
        self.current_value = max_value

    def step(self):
        if self.total_steps <= 1:
            self.current_value = self.min_value
            return self.current_value

        progress = min(self.current_step / self.total_steps, 1.0)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        self.current_value = self.min_value + (self.max_value - self.min_value) * cosine_decay

        self.current_step += 1
        return self.current_value
