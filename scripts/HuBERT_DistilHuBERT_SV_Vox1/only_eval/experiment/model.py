
import torch
from transformers import AutoConfig, HubertModel

from exp_lib import module, augment
from exp_lib.util import TorchModuleManager

class Model(TorchModuleManager):
    def __init__(self, args):
        super(Model, self).__init__()

        # model
        student = module.ssl.StudentHubert_DistHubt(
            args['student_hidden_layer_num'],
            args['student_hidden_layer_size'],
            os_kdft_adapter=args['adapter_size'],
            init_teacher_param=args['init_teacher_idx']
        )

        backend = module.ssl_backend.LinearClassifier(
            args['student_hidden_layer_num'],
            args['student_hidden_layer_size'],
            args['embed_size'], 
            weighted_sum=args['weighted_sum'],
            use_TFT=args['use_TFT']
        )

        self.add_module('student', student)
        self.add_module('backend', backend)

    def __call__(self, x):
        '''Test inference 
        '''
        assert self.state == 'eval', 'do model.eval() first'
        x = self.modules['student'](x)
        x = self.modules['backend'](x)
        return x

    def _load_state_dict(self, student_path, backend_path=None):
        self.modules['student'].load_state_dict(torch.load(student_path), strict=False)
        if backend_path is not None:
            self.modules['backend'].load_state_dict(torch.load(backend_path), strict=False)