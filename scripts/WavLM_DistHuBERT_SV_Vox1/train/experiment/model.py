
import torch
from transformers import AutoConfig, WavLMModel

from exp_lib import module, augment
from exp_lib.util import TorchModuleManager

class Model(TorchModuleManager):
    def __init__(self, args):
        super(Model, self).__init__()
        
        # torch modules
        teacher = WavLMModel.from_pretrained(
            args['huggingface_url'],
            from_tf=bool(".ckpt" in args['huggingface_url']),
            config=AutoConfig.from_pretrained(args['huggingface_url']),
            revision="main",
            ignore_mismatched_sizes=False,
        )

        student = module.ssl.StudentWavLMPlus_DistHubt(
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

        self.add_module('teacher', teacher)
        self.add_module('student', student)
        self.add_module('backend', backend)

    def tune(self, x_kd, x_ft, ft_label):
        '''Perform one iteration learning
        '''
        assert self.state == 'train', 'do model.train() first'
        
        self.optimizer.zero_grad()
        
        # teadcher model inference
        with torch.set_grad_enabled(False):
            kd_label = self.modules['teacher'](x_kd.clone(), output_hidden_states=True).hidden_states
            kd_label = torch.stack(kd_label, dim=1)

        # student model inference
        x = torch.cat((x_kd, x_ft), dim=0)
        x = self.modules['student'](x, idx_without_adapter=x_kd.size(0))
        y_kd, x_ft = x[:x.size(0) // 2], x[x.size(0) // 2:]
        
        # backend model inference
        y_ft = self.modules['backend'](x_ft)

        # calculate loss
        kd_loss = self.modules['kd_loss'](y_kd, kd_label)
        ft_loss = self.modules['ft_loss'](y_ft, ft_label)
        loss = kd_loss + ft_loss

        # back-propagation
        loss.backward()
        self.optimizer.step()

        return kd_loss.item(), ft_loss.item()

    def __call__(self, x):
        '''Test inference 
        '''
        assert self.state == 'eval', 'do model.eval() first'
        x = self.modules['student'](x)
        x = self.modules['backend'](x)
        return x
    
    def set_optimizer(self, optimizer):
        self.optimizer = optimizer
    
    def set_kd_criterion(self, criterion):
        self.add_module('kd_loss', criterion)
    
    def set_ft_criterion(self, criterion):
        self.add_module('ft_loss', criterion)