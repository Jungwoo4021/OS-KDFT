import os
import itertools

def get_args():
    """
    Returns
        system_args (dict): path, log setting
        experiment_args (dict): hyper-parameters
        args (dict): system_args + experiment_args
    """
    system_args = {
        # expeirment info
        'project'                   : 'OS-KDFT',
        'name'                      : 'SV6. OS2_DistHubt_HuBERT_Vox1',
        'tags'                      : ['SV', 'DistHubt', 'HuBERT'],
        'description'               : '',

        # log
        'path_log'                  : '/results',
        'neptune_user'              : 'IRLAB',
        'neptune_token'             : 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIzNmQyNDk2NS0wYWFiLTQxZDEtYjFmOC0xNDIxM2U1YTlmOGQifQ==',
        'wandb_group'               : 'SSL_KDFT',
        'wandb_entity'              : 'ir-lab',
        'wandb_api_key'             : '398ae091e9c781c2974ce0d7c22e09d953920e16',

        # dataset
        'path_libri'                : '/data/ssd2/LibriSpeech_100h',
        'path_train'                : '/data/VoxCeleb1/train',
        'path_test'                 : '/data/VoxCeleb1',
        'path_trials'               : '/data/VoxCeleb1/trials',

        # others
        'num_workers'               : 4,
        'usable_gpu'                : None,
    }

    experiment_args = {
        # huggingface model
        'huggingface_url'           : 'facebook/hubert-base-ls960',
        
        # experiment
        'epoch'                     : 100,
        'batch_size'                : 50,
        'rand_seed'                 : 1,
        
        # model
        'student_hidden_layer_size' : 768,
        'student_hidden_layer_num'  : 2,
        'init_teacher_idx'          : [0, 1],
        'embed_size'                : 192,
        'weighted_sum'              : False,
        'use_TFT'                   : True,
        
        # criterion
        'cos_lambda'                : 1, 
        'target_layer_idx'          : [4, 8, 12], 
        
        # data processing
        'crop_size'                 : 16000 * 3,
        'seg_size'                  : 16000 * 3,
        'num_seg'                   : 5,
    
        # learning rate
        'lr'                        : 5e-5,
        'lr_min'                    : 5e-5,
		'weight_decay'              : 1e-4,
        'T_mult'                    : 1,
    }

    args = {}
    for k, v in itertools.chain(system_args.items(), experiment_args.items()):
        args[k] = v
    args['path_scripts'] = os.path.dirname(os.path.realpath(__file__))
    args['path_log'] = os.path.join(args['path_log'], args['project'], args['name'])

    return args, system_args, experiment_args