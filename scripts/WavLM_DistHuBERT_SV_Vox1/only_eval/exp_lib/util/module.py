import copy
import torch

class TorchModuleManager():
    def __init__(self, **modules):
        self.state = None
        self.device = 'cpu'
        self.modules = {}
        for k, v in modules:
            self.modules[k] = v
            
    def add_module(self, name, model, trainable=True):
        self.modules[name] = model
        self.set_module_trainability(name, trainable)
    
    def set_module_trainability(self, name, trainable):
        for param in self.modules[name].parameters():
            param.requires_grad=trainable
        if not trainable:
            self.modules[name].eval()
    
    def use_distributed_data_parallel(self, device, find_unused_parameters=False):
        for key in self.modules.keys():
            self.modules[key].to(device)

            if self.is_trainable(key):
                self.modules[key] = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.modules[key])
                self.modules[key] = torch.nn.parallel.DistributedDataParallel(
                        self.modules[key], device_ids=[device], find_unused_parameters=find_unused_parameters)

        self.device = device
        self.ddp = True
    
    def compile(self, mode='default'):
        for key in self.modules.keys():
            if self.is_trainable(key):
                self.modules[key] = torch.compile(self.modules[key], mode=mode)
        torch._dynamo.config.verbose=True
        torch._dynamo.config.suppress_errors = True

    def get_parameters(self):
        params = []
        for key, model in self.modules.items():
            if self.is_trainable(key):
                params += list(model.parameters())
        return params
    
    def copy_state_dict(self):
        output = {}
        for key, model in self.modules.items():
            if 0 < len(model.state_dict().keys()):
                output[key] = copy.deepcopy(model.state_dict())
            
        return output
        
    def load_state_dict(self, state_dict):
        for key, params in state_dict.items():
            self.modules[key].load_state_dict(params)

    def eval(self):
        self.state = 'eval'
        for key in self.modules.keys():
            if self.is_trainable(key):
                self.modules[key].eval()
            
    def train(self):
        self.state = 'train'
        for key in self.modules.keys():
            if self.is_trainable(key):
                self.modules[key].train()
            
    def is_trainable(self, key):
        flag_param_exist = 0 < len(self.modules[key].state_dict().keys())
        flag_require_grad = False
        if flag_param_exist:
            for param in self.modules[key].parameters():
                flag_require_grad = param.requires_grad
                if flag_require_grad:
                    break
        return flag_param_exist and flag_require_grad