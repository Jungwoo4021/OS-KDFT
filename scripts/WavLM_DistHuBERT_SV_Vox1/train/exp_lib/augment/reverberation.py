import os
import random
import numpy as np
import soundfile as sf

import torch
import torchaudio

from scipy import signal

class RIRReverberation:
    '''Adjust reverberation using the RIR data set.
    '''
    def __init__(self, path, shared_memory=None, device='cpu'):
        self.files = []
        
        # parse list
        for root, _, files in os.walk(path):
            for file in files:
                if '.wav' in file:
                    self.files.append(os.path.join(root, file))
        
        self.device = device
        self.shared_memory = shared_memory

    def __call__(self, x):
        if self.device == 'cpu':
            return self.cpu_process(x)
        else:
            return self.gpu_process(x)
    
    def cpu_process(self, x):
        path = random.sample(self.files, 1)[0]
        
        if self.shared_memory is None:
            rir, _ = sf.read(path)
        else:
            rir = self.shared_memory[path].copy()
        rir = np.expand_dims(rir, 0)
        rir = rir / np.sqrt(np.sum(rir**2))
        
        x = np.expand_dims(x, 0)
        x = signal.convolve(x, rir, mode='full')[:,:len(x[0])]

        x = np.squeeze(x, 0)

        return x

    def gpu_process(self, x):
        path = random.sample(self.files, 1)[0]
        
        if self.shared_memory is None:
            rir, _ = sf.read(path)
        else:
            rir = self.shared_memory[path].copy()
        rir = torch.from_numpy(rir, device=self.device)
        rir = rir / torch.sqrt(torch.sum(rir**2))

        x = torchaudio.functional.convolve(x, rir.unsqueeze(1), mode='full')

        return x