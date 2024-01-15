import copy
import numpy as np
import random
import soundfile as sf

from torch.utils.data import Dataset, DataLoader, DistributedSampler

from data_processing import Musan, RIRReverberation, audio_clipping, EasingAugmentation

def get_loaders(args, vox):
    train_set = TrainSet(args, vox.train_set)
    train_sampler = DistributedSampler(train_set, shuffle=True)
    train_loader = DataLoader(
        train_set,
        num_workers=args['num_workers'],
        batch_size=args['batch_size'],
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True
    )
    
    test_set = EnrollmentSet(args, vox.test_set_O)
    test_sampler = DistributedSampler(test_set, shuffle=False)
    test_loader_O = DataLoader(
        test_set,
        num_workers=args['num_workers'],
        batch_size=1,
        sampler=test_sampler,
        pin_memory=True,
    )

    test_set = EnrollmentSet(args, vox.test_set_E)
    test_sampler = DistributedSampler(test_set, shuffle=False)
    test_loader_E = DataLoader(
        test_set,
        num_workers=args['num_workers'] * 2,
        batch_size=1,
        sampler=test_sampler,
        pin_memory=True,
    )

    return train_set, train_sampler, train_loader, test_loader_O, test_loader_E

class TrainSet(Dataset):
    def __init__(self, args, items):
        self.items = items
        self.musan = Musan(args['path_musan'])
        self.rir = RIRReverberation(args['path_rir'])
        self.ease = EasingAugmentation()
        self.crop_size = args['num_train_frames'] * 160
        self.p = args['DA_p']

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        # sample
        item = self.items[index]
        
        # read wav
        info = sf.info(item.path)

        start = random.randint(0, int(info.duration - (self.crop_size / info.samplerate)))
        audio, _ = sf.read(item.path, start=start * info.samplerate, stop=start * info.samplerate + self.crop_size)
        
        audio1 = copy.deepcopy(audio)
        audio2 = self.augment(audio, self.p)
        
        return audio1.astype(np.float), audio2.astype(np.float), item.label

    def augment(self, audio, p):
        if p <  random.random():
            return audio

        # Musan & RIR
        aug_type = random.randint(0, 5)
        if aug_type == 0:
            pass
        elif aug_type == 1:
            audio = self.rir(audio)
        elif aug_type == 2:
            audio = self.musan(audio, 'speech')
        elif aug_type == 3:
            audio = self.musan(audio, 'music')
        elif aug_type == 4:
            audio = self.musan(audio, 'noise')
        elif aug_type == 5:
            audio = self.musan(audio, 'speech')
            audio = self.musan(audio, 'music')

        # clip
        if random.random() < 0.3:
            audio = audio_clipping(audio, (0.5, 0.9))
            
        # easing
        if random.random() < 0.3:
            audio = self.ease(audio)

        return audio

class EnrollmentSet(Dataset):
    def __init__(self, args, items):
        self.items = items
        self.crop_size = args['num_test_frames'] * 160
        self.num_seg = args['num_seg']
    
    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):        
        # sample
        item = self.items[index]

        # read wav
        audio, _ = sf.read(item.path)
        
        # full utterance
        full = audio.copy()

        # crop
        if audio.shape[0] <= self.crop_size:
            shortage = self.crop_size - audio.shape[0]
            audio = np.pad(audio, (0, shortage), 'wrap')

        # stack
        buffer = []
        indices = np.linspace(0, audio.shape[0] - self.crop_size, self.num_seg)
        for idx in indices:
            idx = int(idx)
            buffer.append(audio[idx:idx + self.crop_size])
        buffer = np.stack(buffer, axis=0)

        return full.astype(np.float), buffer.astype(np.float), item.key