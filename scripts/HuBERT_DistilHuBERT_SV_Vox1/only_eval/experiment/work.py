from tqdm import tqdm
from itertools import cycle

import torch
import torch.distributed as dist

from exp_lib import test

def val(model, data_loader_TTA, trials):
    model.eval()
    
    # enrollment
    embeddings_TTA = test.SV_enrollment(model, data_loader_TTA, use_TTA=True, run_on_ddp=True)

    # EER
    eer = test.test_SV_EER(trials, multi_embedding=embeddings_TTA)
    _synchronize()

    return eer

def eval(model, data_loader, data_loader_TTA, trials):
    model.eval()
    
    # enrollment
    embeddings_full = test.SV_enrollment(model, data_loader, use_TTA=False, run_on_ddp=True)
    embeddings_TTA = test.SV_enrollment(model, data_loader_TTA, use_TTA=True, run_on_ddp=True)

    # EER
    eer = test.test_SV_EER(trials, mono_embedding=embeddings_full, multi_embedding=embeddings_TTA)
    _synchronize()

    return eer

def _synchronize():
    torch.cuda.empty_cache()
    dist.barrier()