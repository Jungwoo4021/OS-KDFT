import os
from tqdm import tqdm
import numpy as np
import random
import datetime
from multiprocessing import Manager

import torch
import soundfile as sf

import arguments
import experiment
from exp_lib import log, loss, test

def run(process_id, args, experiment_args):
    #======================================
    #           experiment setup
    #======================================
    
    # set reproducible
    torch.cuda.empty_cache()
    random.seed(args['rand_seed'])
    np.random.seed(args['rand_seed'])
    torch.manual_seed(args['rand_seed'])
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    
    # DDP env
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = args['port']
    args['rank'] = process_id
    args['device'] = f'cuda:{process_id}'
    torch.distributed.init_process_group(
            backend='nccl', world_size=args['world_size'], rank=args['rank'])
           
    # logger
    if process_id == 0:
        builder = log.LogManager.Builder(
            args['name'], args['project'], args['tags'], args['description'], args['path_scripts'], experiment_args
        )
        builder.use_local_logger(args['path_log'])
        #builder.use_neptune_logger(args['neptune_user'], args['neptune_token'])
        #builder.use_wandb_logger(args['wandb_entity'], args['wandb_api_key'], args['wandb_group'])
        logger = builder.build()
        logger.log_arguments(experiment_args)
    else:
        logger = None
        
    # data
    data_manager = experiment.DataManager(args)
        
    # criterion
    kd_criterion = loss.DistilHubertKDLoss(
        args['cos_lambda'], 
        args['target_layer_idx'], 
        args['student_hidden_layer_size'] 
    )
    ft_criterion = loss.CCE(
        data_manager.vox1.NUM_TRAIN_SPK,
        args['embed_size'],
        data_manager.vox1.class_weight
    )
    
    # model
    model = experiment.Model(args)
    model.set_module_trainability('teacher', False)
    model.set_kd_criterion(kd_criterion)
    model.set_ft_criterion(ft_criterion)
    model.use_distributed_data_parallel(args['device'], find_unused_parameters=True)

    # optimizer
    optimizer = torch.optim.Adam(
        model.get_parameters(),
        lr=args['lr'], 
        weight_decay=args['weight_decay']
    )
    model.set_optimizer(optimizer)

    # lr scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=args['epoch'],
        T_mult=args['T_mult'],
        eta_min=args['lr_min']
    )
    


    #==========================================
    #              run experiment
    #==========================================
    best_eer = 100
    for epoch in range(1, args['epoch'] + 1):
        scheduler.step(epoch)
        data_manager.set_epoch(epoch)
        
        # train
        experiment.train(
            current_epoch=epoch,
            model=model,
            data_loader=data_manager.train_loader,
            logger=logger
        )

        # validation
        eer = experiment.val(
            model, 
            data_loader_TTA=data_manager.test_loader_O_TTA, 
            trials=data_manager.vox1.trials
        )

        if eer < best_eer:
            best_eer = eer
            best_state = model.copy_state_dict()
            if logger is not None:
                for k, v in best_state.items():
                    logger.save_model(f'FT_checkpoint[{epoch}]_{k}', v)
        if logger is not None:
            logger.log_metric('eer/val', eer, epoch)

    # evaluation
    model.load_state_dict(best_state)
    eer = experiment.eval(
        model, 
        data_loader=data_manager.test_loader_O, 
        data_loader_TTA=data_manager.test_loader_O_TTA, 
        trials=data_manager.vox1.trials
    )
    if logger is not None:
        logger.log_metric('eer/eval', eer, epoch)
        
if __name__ == '__main__':
    # get arguments
    args, system_args, experiment_args = arguments.get_args()
    
    # check gpu environment
    if args['usable_gpu'] is None: 
        args['gpu_ids'] = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = args['usable_gpu']
        args['gpu_ids'] = args['usable_gpu'].split(',')
    assert 0 < len(args['gpu_ids']), 'Only GPU env are supported'

    # set DDP environment
    args['port'] = f'4{datetime.datetime.now().microsecond % 1000}'
    args['world_size'] = len(args['gpu_ids'])
    args['batch_size'] = args['batch_size'] // args['world_size']
    
    # start
    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.multiprocessing.spawn(
        run, 
        nprocs=args['world_size'], 
        args=(args, experiment_args)
    )