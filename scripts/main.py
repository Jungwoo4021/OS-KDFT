import os
import random
import datetime
import numpy as np

import torch
import torch.nn as nn

import arguments
import data_loaders
import train
from dataset import VoxCeleb2
from logger import Logger
from transformers import AutoConfig, Wav2Vec2Model
from models.w2v_ecapa import W2V2_ECAPA
from models.student_w2v2 import Wav2Vec2_Mini

def set_experiment_environment(args):
    # reproducible
    random.seed(args['rand_seed'])
    np.random.seed(args['rand_seed'])
    torch.manual_seed(args['rand_seed'])
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    # DDP env
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = args['port']
    args['rank'] = args['process_id']
    args['device'] = f'cuda:{args["process_id"]}'
    torch.distributed.init_process_group(
            backend='nccl', world_size=args['world_size'], rank=args['rank'])

def run(process_id, args, experiment_args):
    # check parent process
    args['process_id'] = process_id
    args['flag_parent'] = process_id == 0
    
    # args
    set_experiment_environment(args)
    trainer = train.ModelTrainer()
    trainer.args = args

    # logger
    if args['flag_parent']:
        trainer.logger = Logger.Builder(args['name'], args['project']
                ).tags(args['tags']
                ).description(args['description']
                ).save_source_files(args['path_scripts']
                ).use_local(args['path_log']
                #).use_wandb(args['wandb_group'], args['wandb_entity']
                ).use_neptune(args['neptune_user'], args['neptune_token']
                ).build()
        trainer.logger.log_parameter(experiment_args)

    # voxceleb
    trainer.vox = VoxCeleb2(args['path_train'], args['path_test'], args['path_trials'])

    # data loader
    trainer.train_set, trainer.train_sampler, trainer.train_loader, trainer.test_loader_O, trainer.test_loader_E = data_loaders.get_loaders(args, trainer.vox)

    # Wav2Vec2 - teacher
    config = AutoConfig.from_pretrained(
        args['model_name_or_path'],
        finetuning_task="audio-classification",
        revision="main",
    )

    trainer.wav2vec_t = Wav2Vec2Model.from_pretrained(
        args['model_name_or_path'],
        from_tf=bool(".ckpt" in args['model_name_or_path']),
        config=config,
        revision="main",
        ignore_mismatched_sizes=False,
    )
    trainer.wav2vec_t = trainer.wav2vec_t.to(args['device'])
    for param in trainer.wav2vec_t.parameters():
        param.requires_grad = False
    
    # Wav2Vec2 - student
    config = AutoConfig.from_pretrained(
        args['model_name_or_path'],
        num_labels=args['n_class'],
        num_hidden_layers=args['num_hidden_layers'],
        finetuning_task="audio-classification",
        revision="main",
    )

    trainer.wav2vec_s = Wav2Vec2_Mini(config=config)
    trainer.wav2vec_s.feature_extractor.load_state_dict(trainer.wav2vec_t.feature_extractor.state_dict(), strict=False)
    trainer.wav2vec_s.feature_projection.load_state_dict(trainer.wav2vec_t.feature_projection.state_dict(), strict=False)
    for i in range(args['num_hidden_layers']):
        trainer.wav2vec_s.encoder.layers[i].load_state_dict(trainer.wav2vec_t.encoder.layers[i].state_dict(), strict=False)
    trainer.wav2vec_s = trainer.wav2vec_s.to(args['device'])
    trainer.wav2vec_s = nn.SyncBatchNorm.convert_sync_batchnorm(trainer.wav2vec_s)
    trainer.wav2vec_s = nn.parallel.DistributedDataParallel(
        trainer.wav2vec_s, device_ids=[args['device']], find_unused_parameters=True
    )

    # ECAPA
    trainer.ecapa = W2V2_ECAPA(args, torch.FloatTensor(trainer.vox.class_weight))
    trainer.ecapa = trainer.ecapa.to(args['device'])
    trainer.ecapa = nn.SyncBatchNorm.convert_sync_batchnorm(trainer.ecapa)
    trainer.ecapa = nn.parallel.DistributedDataParallel(
        trainer.ecapa, device_ids=[args['device']], find_unused_parameters=False
    )

    # KD Loss
    trainer.loss_KD = nn.MSELoss().to(args['device'])

    # optimizer
    trainer.lr_guide = torch.optim.Adam(
        list(trainer.wav2vec_s.parameters()) + list(trainer.ecapa.parameters()), 
        lr=args['lr'], 
        weight_decay=args['weight_decay']
    )

    # lr scheduler
    trainer.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        trainer.lr_guide,
        T_0=args['T_0'],
        T_mult=args['T_mult'],
        eta_min=args['lr_min']
    )

    trainer.run()

if __name__ == '__main__':
    # get arguments
    args, system_args, experiment_args = arguments.get_args()
    
    # set reproducible
    random.seed(args['rand_seed'])
    np.random.seed(args['rand_seed'])
    torch.manual_seed(args['rand_seed'])

    # set gpu device
    if args['usable_gpu'] is None: 
        args['gpu_ids'] = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = args['usable_gpu']
        args['gpu_ids'] = args['usable_gpu'].split(',')
    
    args['port'] = f'10{datetime.datetime.now().microsecond % 100}'
    assert 0 < len(args['gpu_ids']), 'Only GPU env are supported'

    # set DDP
    args['world_size'] = len(args['gpu_ids'])
    args['batch_size'] = args['batch_size'] // args['world_size']
    
    # start
    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.multiprocessing.spawn(
        run, 
        nprocs=args['world_size'], 
        args=(args, experiment_args)
    )