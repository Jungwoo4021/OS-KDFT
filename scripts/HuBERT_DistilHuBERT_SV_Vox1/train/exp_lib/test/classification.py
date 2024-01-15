import os
from tqdm import tqdm

import torch

from ..util import all_gather, synchronize

def calculate_accuracy(model, loader, num_class, run_on_ddp=False):
    # set test mode
    model.eval()

    corrects = [0 for _ in range(num_class)]
    total_samples = [0 for _ in range(num_class)]
    predicts = [0 for _ in range(num_class)]

    with torch.set_grad_enabled(False):
        for x, label in tqdm(loader, desc='calculate acc(%)', ncols=90):                
            # to cuda
            if run_on_ddp:
                x = x.to(dtype=torch.float32, device=model.device, non_blocking=True)
            else:
                x = x.to(dtype=torch.float32, device=model.device)
            
            # inference
            x = model(x)
            p = torch.nn.functional.softmax(x, dim=-1)
            p = torch.max(p, dim=1)[1]
            
            # count
            for i in range(p.size(0)):
                _p = p[i].item()
                l = label[i].item()
                predicts[_p] += 1
                total_samples[l] += 1
                if _p == l:
                    corrects[l] += 1
    
    if run_on_ddp:
        synchronize()
        corrects = all_gather(corrects)
        total_samples = all_gather(total_samples)
        return sum(corrects) / sum(total_samples) * 100
    else:
        return sum(corrects) / sum(total_samples) * 100