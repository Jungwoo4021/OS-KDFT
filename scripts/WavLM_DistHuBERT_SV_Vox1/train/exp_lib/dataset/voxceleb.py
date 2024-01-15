import os

from ._dataclass import SV_Trial, SV_TrainItem, SV_EnrollmentItem, SID_TrainItem, SID_TestItem

class VoxCeleb1:
    NUM_TRAIN_ITEM = 148642
    NUM_TRAIN_SPK = 1211
    NUM_TRIALS = 37611

    def __init__(self, path_train, path_test, path_trials):
        self.train_set = []
        self.test_set = []
        self.trials = []
        self.class_weight = []

        # train_set
        labels = {}
        num_utt = [0 for _ in range(self.NUM_TRAIN_SPK)]
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
                    item = SV_TrainItem(path=f, speaker=spk, label=labels[spk])
                    self.train_set.append(item)
                    num_sample += 1
                    num_utt[labels[spk]] += 1

        for n in num_utt:
            self.class_weight.append(num_sample / n)
                    
        # test_set
        for root, _, files in os.walk(os.path.join(path_test, 'test')):
            for file in files:
                if '.wav' in file:
                    f = os.path.join(root, file)
                    item = SV_EnrollmentItem(path=f, key='/'.join(f.split('/')[-3:]))
                    self.test_set.append(item)

        self.trials = self.parse_trials(os.path.join(path_trials, 'trials.txt'))

        # error check
        assert len(self.train_set) == self.NUM_TRAIN_ITEM
        assert len(self.trials) == self.NUM_TRIALS
        assert len(labels) == self.NUM_TRAIN_SPK

    def parse_trials(self, path):
        trials = []
        for line in open(path).readlines():
            strI = line.split(' ')
            item = SV_Trial(
                key1=strI[1].replace('\n', ''), 
                key2=strI[2].replace('\n', ''), 
                label=int(strI[0])
            )
            trials.append(item)
        return trials

class VoxCeleb2:
    NUM_TRAIN_ITEM = 1092009
    NUM_TRAIN_SPK = 5994
    NUM_TRIALS = 37611
    NUM_TRIALS_E = 579818
    NUM_TRIALS_H = 550894

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
        num_utt = [0 for _ in range(self.NUM_TRAIN_SPK)]
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
                    item = SV_TrainItem(path=f, speaker=spk, label=labels[spk])
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
                    item = SV_EnrollmentItem(path=f, key='/'.join(f.split('/')[-3:]))
                    self.test_set_O.append(item)

        # test_set_E
        for root, _, files in os.walk(os.path.join(path_test)):
            for file in files:
                if '.wav' in file:
                    f = os.path.join(root, file)
                    item = SV_EnrollmentItem(path=f, key='/'.join(f.split('/')[-3:]))
                    self.test_set_E.append(item)

        
        self.trials_O = self.parse_trials(os.path.join(path_trials, 'trials.txt'))
        self.trials_E = self.parse_trials(os.path.join(path_trials, 'trials_E.txt'))
        self.trials_H = self.parse_trials(os.path.join(path_trials, 'trials_H.txt'))

        # error check
        assert len(self.train_set) == self.NUM_TRAIN_ITEM
        assert len(self.trials_O) == self.NUM_TRIALS
        assert len(self.trials_E) == self.NUM_TRIALS_E
        assert len(self.trials_H) == self.NUM_TRIALS_H
        assert len(labels) == self.NUM_TRAIN_SPK

    def parse_trials(self, path):
        trials = []
        for line in open(path).readlines():
            strI = line.split(' ')
            item = SV_Trial(
                key1=strI[1].replace('\n', ''), 
                key2=strI[2].replace('\n', ''), 
                label=int(strI[0])
            )
            trials.append(item)
        return trials
    
class Vox1_SID:
    NUM_TRAIN_ITEM = 145265
    NUM_TRAIN_SPK = 1251
    NUM_TEST_ITEM = 8251
    NUM_TEST_SPK = 1251

    def __init__(self, path_train, path_test):
        self.train_set = []
        self.test_set = []
        self.trials = []
        self.class_weight = []

        # train_set
        labels = {}
        num_utt = [0 for _ in range(self.NUM_TRAIN_SPK)]
        num_train_sample = 0
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
                    item = SID_TrainItem(path=f, speaker=spk, label=labels[spk])
                    self.train_set.append(item)
                    num_train_sample += 1
                    num_utt[labels[spk]] += 1

        for n in num_utt:
            self.class_weight.append(num_train_sample / n)
                    
        # test_set
        test_labels = {}
        num_test_sample = 0
        for root, _, files in os.walk(path_test):
            for file in files:
                if '.wav' in file:
                    f = os.path.join(root, file)
                    
                    # parse speaker
                    spk = f.split('/')[-3]

                    # labeling
                    try: test_labels[spk]
                    except: 
                        test_labels[spk] = len(test_labels.keys())
                    
                    # init item
                    item = SID_TestItem(path=f, speaker=spk, label=test_labels[spk])
                    self.test_set.append(item)
                    num_test_sample += 1

        # error check
        assert len(self.train_set) == self.NUM_TRAIN_ITEM
        assert len(labels) == self.NUM_TRAIN_SPK
        assert len(test_labels) == self.NUM_TEST_SPK
        assert len(self.test_set) == self.NUM_TEST_ITEM