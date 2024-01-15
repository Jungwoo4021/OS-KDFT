from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.distributed as dist
from sklearn.metrics import accuracy_score

from ..util import all_gather, synchronize

def SID_test(framework, loader, run_on_ddp=False):
    '''Test speaker identification performance and return Acc 
    
    Param
        loader: DataLoader that returns (wav, label)
        
    Return
        score(float)
    '''
    framework.eval()
    softmax = torch.nn.Softmax(dim=-1)

    labels = []
    predicts = []
    with tqdm(total=len(loader), ncols=90) as pbar, torch.set_grad_enabled(False):
        for x, label in loader:
            if run_on_ddp:
                x = x.to(torch.float32).to(framework.device, non_blocking=True)
            else:
                x = x.to(torch.float32).to(framework.device)
            
            # feed forward
            x = framework(x).to('cpu')
            pred = softmax(x)
            pred = torch.argmax(pred, dim=-1)
            
            # append
            predicts.append(pred.item())
            labels.append(label.item())
            
            pbar.update(1)
    
    if run_on_ddp:
        synchronize()

        # gather
        predicts = all_gather(predicts)
        labels = all_gather(labels)

        score = accuracy_score(predicts, labels)
        print('Test_acc: ', score * 100)

    return score * 100
