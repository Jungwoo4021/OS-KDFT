import random
import torch

class SpectrogramMasking:
    def __init__(self, F, T, time_first=False):
        self.F = F
        self.T = T
        self.time_first = time_first
    
    def __call__(self, x):
        if self.time_first and len(x.size()) == 3:
            x = x.permute(0, 2, 1)
        if self.time_first and len(x.size()) == 4:
            x = x.permute(0, 1, 3, 2)

        f = random.randint(0, self.F)
        t = random.randint(0, self.T)
        m = torch.ones(x.size(), device=x.device, requires_grad=False)
        idx_f = random.randint(0, m.size(-2) - f)
        idx_t = random.randint(0, m.size(-1) - t)
        if len(x.size()) == 3:
            m[:, idx_f:idx_f + f, idx_t:idx_t + t] = 0    
        elif len(x.size()) == 4:
            m[:, :, idx_f:idx_f + f, idx_t:idx_t + t] = 0    
        x = x * m

        if self.time_first and len(x.size()) == 3:
            x = x.permute(0, 2, 1)
        if self.time_first and len(x.size()) == 4:
            x = x.permute(0, 1, 3, 2)

        return x