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
        'project'       : 'OS_KDFT',
        'name'          : 'OS_KDFT',
        'tags'          : ['proposed'],
        'description'   : '',

        # log
        'path_log'      : '/results',
        'neptune_user'  : '',
        'neptune_token' : '',
        'wandb_group'   : '',
        'wandb_entity'  : '',

        # dataset
        'path_train'    : {YOUR_PATH},
        'path_test'     : {YOUR_PATH},
        'path_trials'  	: {YOUR_PATH},
        'path_musan'  	: {YOUR_PATH},
        'path_rir'      : {YOUR_PATH},

        # others
        'num_workers': 4,
        'usable_gpu': None,
    }

    experiment_args = {
        # huggingface model
        'model_name_or_path'    : 'facebook/wav2vec2-large-xlsr-53',
        
        # experiment
        'epoch'             : 100,
        'batch_size'        : 128,
        'rand_seed'		    : 1,
        
        # model
        'C'                 : 512,
        'num_hidden_layers' : 4,
        'n_class'           : 5994, 
        'embedding_size'	: 192,
        'aam_margin'        : 0.2,
        'aam_scale'         : 30,
        'spec_mask_F'       : 100,
        'spec_mask_T'       : 10,

        # data processing
        'num_seg'           : 5,
        'num_train_frames'  : 200,
        'num_test_frames'   : 300,
        'DA_p'              : 0.6,
    
        # learning rate
        'lr'            : 1e-4,
        'lr_min'        : 1e-6,
        'weight_decay'  : 0,
        'T_0'           : 100,
        'T_mult'        : 1,
    }

    args = {}
    for k, v in itertools.chain(system_args.items(), experiment_args.items()):
        args[k] = v
    args['path_scripts'] = os.path.dirname(os.path.realpath(__file__))

    return args, system_args, experiment_args
