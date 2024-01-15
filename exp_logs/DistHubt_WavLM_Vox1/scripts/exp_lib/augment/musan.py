import os
import random
import numpy as np
import soundfile as sf

class MusanNoiseInjection:
    '''Inject noise using MUSAN dataset. 
    see here -> MUSAN: A Music, Speech, and Noise Corpus (https://arxiv.org/pdf/1510.08484.pdf)
    '''
    def __init__(self, path, category=None, snr=None, num_file=None, shared_memory=None):
        # set vars
        self.category = category 
        if self.category is None:
            self.category = ['noise','speech','music']
        self.snr = snr
        if self.snr is None:
            self.snr = { 'noise': (0, 15), 'speech': (13, 20), 'music': (5, 25) }
        self.num_file = num_file
        if self.num_file is None:
            self.num_file = { 'noise': (1, 1), 'speech': (3, 8), 'music': (1, 1) }

        self.noise_list = {}
        self.num_noise_file = {}

        for category in self.category:
            self.noise_list[category] = []
            self.num_noise_file[category] = 0

        # init noise list
        for root, _, files in os.walk(path):
            if self.category[0] in root:
                category = self.category[0]
            elif self.category[1] in root:
                category = self.category[1]
            elif self.category[2] in root:
                category = self.category[2]

            for file in files:
                if '.wav' in file:
                    self.noise_list[category].append(os.path.join(root, file))
                    self.num_noise_file[category] += 1

        self.shared_memory = shared_memory

    def __call__(self, x, category):
        # calculate dB
        x_size = x.shape[-1]
        x_dB = self.calculate_decibel(x)

        # select noise
        snr_min, snr_max = self.snr[category]
        file_min, file_max = self.num_file[category]
        files = random.sample(
            self.noise_list[category],
            random.randint(file_min, file_max)
        )

        # init noise
        noises = []
        for f in files:
            info = sf.info(f)
            wav_size = int(info.samplerate * info.duration)

            if wav_size <= x_size:
                # load wav
                if self.shared_memory is None:
                    noise, _ = sf.read(f)
                else:
                    noise = self.shared_memory[f].copy()
                # duplicate
                noise_size = noise.shape[0]
                if noise_size < x_size:
                    shortage = x_size - noise_size
                    noise = np.pad(noise, (0, shortage), 'wrap')
            else:
                # load wav
                index = random.randint(0, wav_size - x_size - 1)
                if self.shared_memory is None:
                    noise, _ = sf.read(f, start=index, stop=index + x_size)
                else:
                    noise = self.shared_memory[f].copy()
                    noise = noise[index:index + x_size]

            noises.append(noise)
        noise = np.mean(noises, axis=0)
        
        # set SNR
        snr = random.uniform(snr_min, snr_max)
        
        # calculate dB
        noise_dB = self.calculate_decibel(noise)
        
        # inject
        p = (x_dB - noise_dB - snr)
        x += np.sqrt(10 ** (p / 10)) * noise
        
        return x

    def calculate_decibel(self, wav):
        assert 0 <= np.mean(wav ** 2) + 1e-4 
        return 10 * np.log10(np.mean(wav ** 2) + 1e-4)