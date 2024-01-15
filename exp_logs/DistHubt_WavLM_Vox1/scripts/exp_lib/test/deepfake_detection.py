from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.distributed as dist

from .metric import calculate_EER
from ..util import all_gather, synchronize

def DF_test(framework, loader, run_on_ddp=False, get_scores=False):
    '''Test deepfake detection performance and return EER 
    
    Param
        loader: DataLoader that returns (wav, label)
        get_scores: if True, returns the score and label used in EER calculation.
        
    Return
        eer(float)
        score(list(float))
        labels(list(int))
    '''
    framework.eval()

    labels = []
    scores = []
    with tqdm(total=len(loader), ncols=90) as pbar, torch.set_grad_enabled(False):
        for x, label in loader:
            if run_on_ddp:
                x = x.to(torch.float32).to(framework.device, non_blocking=True)
            else:
                x = x.to(torch.float32).to(framework.device)
            x = framework(x).to('cpu')
            
            for i in range(x.size(0)):
                if x.size(-1) == 1:
                    scores.append(x[i, 0].item())
                elif x.size(-1) == 2:
                    scores.append(x[i, 1].item())
                else:
                    raise NotImplemented
                labels.append(label[i].item())
        
            pbar.update(1)
    
    if run_on_ddp:
        torch.cuda.empty_cache()
        dist.barrier()
        scores = all_gather(scores)
        labels = all_gather(labels)

    eer = calculate_EER(scores, labels)
    
    if get_scores:
        return eer, scores, labels
    else:
        return eer