import os
from dataclasses import dataclass

NUM_TRAIN_ITEM = 1092009 * 3
NUM_TRAIN_SPK = 5994 * 3
NUM_TRIALS = 37611
NUM_TRIALS_E = 579818
NUM_TRIALS_H = 550894

@dataclass
class TrainItem:
    path: str
    speaker: str
    label: int

@dataclass
class EnrollmentItem:
    key: str
    path: str

@dataclass
class Trial:
    key1: str
    key2: str
    label: int

class VoxCeleb2_TimeStretch:
    def __init__(self, path_train, path_test, path_trials):
        self.train_set = []
        self.test_set_O = []
        self.test_set_E = []
        self.trials_O = []
        self.trials_H = []
        self.trials_E = []
        self.class_weight = []

        # train_set
        labels = {}
        num_utt = [0 for _ in range(NUM_TRAIN_SPK)]
        num_sample = 0
        for root, _, files in os.walk(path_train):
            for file in files:
                if '.wav' in file:
                    # combine path
                    f = os.path.join(root, file)
                    
                    # parse speaker
                    spk = f.split('/')[-3]
                    
                    # labeling
                    try: labels[spk]
                    except: 
                        labels[spk] = len(labels.keys())

                    # init item
                    item = TrainItem(path=f, speaker=spk, label=labels[spk])
                    self.train_set.append(item)
                    num_sample += 1
                    num_utt[labels[spk]] += 1

        for n in num_utt:
            self.class_weight.append(num_sample / n)
                    
        # test_set_O
        for root, _, files in os.walk(os.path.join(path_test, 'test')):
            for file in files:
                if '.wav' in file:
                    f = os.path.join(root, file)
                    item = EnrollmentItem(path=f, key='/'.join(f.split('/')[-3:]))
                    self.test_set_O.append(item)

        # test_set_E
        for root, _, files in os.walk(os.path.join(path_test, 'train')):
            for file in files:
                if '.wav' in file:
                    f = os.path.join(root, file)
                    item = EnrollmentItem(path=f, key='/'.join(f.split('/')[-3:]))
                    self.test_set_E.append(item)

        
        self.trials_O = self.parse_trials(os.path.join(path_trials, 'trials.txt'))
        self.trials_E = self.parse_trials(os.path.join(path_trials, 'trials_E.txt'))
        self.trials_H = self.parse_trials(os.path.join(path_trials, 'trials_H.txt'))

        # error check
        assert len(self.train_set) == NUM_TRAIN_ITEM
        assert len(self.trials_O) == NUM_TRIALS
        assert len(self.trials_E) == NUM_TRIALS_E
        assert len(self.trials_H) == NUM_TRIALS_H
        assert len(labels) == NUM_TRAIN_SPK

    def parse_trials(self, path):
        trials = []
        for line in open(path).readlines():
            strI = line.split(' ')
            item = Trial(
                key1=strI[1].replace('\n', ''), 
                key2=strI[2].replace('\n', ''), 
                label=int(strI[0])
            )
            trials.append(item)
        return trials