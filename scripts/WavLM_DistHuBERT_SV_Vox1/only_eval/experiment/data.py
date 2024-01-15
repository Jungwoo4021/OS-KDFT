import random
import numpy as np
import soundfile as sf

from torch.utils.data import Dataset, DataLoader, DistributedSampler

from exp_lib import util, dataset, augment

class DataManager():
    '''Initialize and control all datasets & dataloaders used in the experiment. 
    '''
    def __init__(self, args):
        self.total_epoch = args['epoch']
        self.vox1 = dataset.VoxCeleb1(args['path_train'], args['path_test'], args['path_trials'])
        
        # test loader
        self.test_set_O = EnrollmentSet(self.vox1.test_set)
        self.test_loader_O = DataLoader(
            self.test_set_O,
            num_workers=args['num_workers'] * 2,
            batch_size=1,
            sampler=DistributedSampler(self.test_set_O, shuffle=False),
            pin_memory=True,
        )
        
        self.test_set_O_TTA = TTAEnrollmentSet(self.vox1.test_set, args['num_seg'], args['seg_size'])
        self.test_loader_O_TTA = DataLoader(
            self.test_set_O_TTA,
            num_workers=args['num_workers'],
            batch_size=args['batch_size'] // args['num_seg'],
            sampler=DistributedSampler(self.test_set_O_TTA, shuffle=False),
            pin_memory=True,
        )

class EnrollmentSet(Dataset):
    def __init__(self, items):
        self.items = items
    
    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):        
        # sample
        item = self.items[index]

        # read wav
        wav, _ = sf.read(item.path)
        
        return wav, item.key
    
class TTAEnrollmentSet(Dataset):
    def __init__(self, items, num_seg, seg_size):
        self.items = items
        self.num_seg = num_seg
        self.seg_size = seg_size
    
    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):        
        # sample
        item = self.items[index]

        # read wav
        wav_TTA = util.linspace_crop_read(item.path, self.num_seg, self.seg_size)
        
        return wav_TTA, item.key