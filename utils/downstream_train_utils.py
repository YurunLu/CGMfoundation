import torch, numpy as np

class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, max_steps, min_lr=1e-6, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1
        if self.warmup_steps > 0 and step < self.warmup_steps:
            scale = step / self.warmup_steps
        else:
            # progress = (step - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps)
            # 加一个 min(1.0, ...) 强制截断，超过 max_steps 后永远当做 1.0 处理
            progress = min(1.0, (step - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps))
            scale = 0.5 * (1.0 + np.cos(np.pi * progress))
        return [self.min_lr + scale * (base_lr - self.min_lr) for base_lr in self.base_lrs]
