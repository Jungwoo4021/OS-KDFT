from tqdm import tqdm
from itertools import cycle

import torch
import torch.distributed as dist

from exp_lib import test

def train(current_epoch, model, data_loader, logger):
    model.train()
    
    count = 0
    kd_loss_sum = 0
    ft_loss_sum = 0

    with tqdm(total=len(data_loader), ncols=90) as pbar:
        for x_kd, x_ft, ft_label in data_loader:
            # to GPU
            x_kd = x_kd.to(dtype=torch.float32, device=model.device)
            x_ft = x_ft.to(dtype=torch.float32, device=model.device)
            ft_label = ft_label.to(dtype=torch.int64, device=model.device)

            # train
            kd_loss, ft_loss = model.tune(x_kd, x_ft, ft_label)
            
            # logging
            if logger is not None:
                count += 1
                kd_loss_sum += kd_loss
                ft_loss_sum += ft_loss
                if len(data_loader) * 0.02 <= count:
                    logger.log_metric('Loss/KD', kd_loss_sum / count)
                    logger.log_metric('Loss/FT', ft_loss_sum / count)
                    kd_loss_sum = 0
                    ft_loss_sum = 0
                    count = 0

                desc = f'train_FT-[{current_epoch}|(loss): {kd_loss + ft_loss:.4f}'
                pbar.set_description(desc)
                pbar.update(1)

    _synchronize()

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