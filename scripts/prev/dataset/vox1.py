import os
from dataclasses import dataclass

NUM_TRAIN_ITEM = 148642
NUM_TRAIN_SPK = 1211
NUM_TRIALS = 37611
NUM_TEST_ITEM = 4874

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

class VoxCeleb1:
    def __init__(self, path_train, path_test, path_trials):
        self.train_set = []
        self.test_set = []
        self.trials = []
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
                    
        # test_set
        for root, _, files in os.walk(path_test):
            for file in files:
                if '.wav' in file:
                    f = os.path.join(root, file)
                    item = EnrollmentItem(path=f, key='/'.join(f.split('/')[-3:]))
                    self.test_set.append(item)

        # trials
        for line in open(os.path.join(path_trials, 'trials.txt')).readlines():
            strI = line.split(' ')
            item = Trial(
                key1=strI[1].replace('\n', ''), 
                key2=strI[2].replace('\n', ''), 
                label=int(strI[0])
            )
            self.trials.append(item)

        # error check
        assert len(self.train_set) == NUM_TRAIN_ITEM
        assert len(self.test_set) == NUM_TEST_ITEM
        assert len(self.trials) == NUM_TRIALS
        assert len(labels) == NUM_TRAIN_SPK