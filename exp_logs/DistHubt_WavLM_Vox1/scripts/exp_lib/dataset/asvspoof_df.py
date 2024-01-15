import os

from ._dataclass import DF_Item

class ASVspoof2021_DF:
    NUM_TEST_ITEM   = 611829

    # HS fix here
    def __init__(self, path_train, path_test, path_train_trl, path_test_trl):
        self.train_set = []
        self.test_set = []
        self.class_weight = []

        # train_set
        train_num_pos = 0
        train_num_neg = 0
        
        trl = os.path.join(path_train_trl)
        for line in open(trl).readlines():
            strI = line.replace('\n', '').split(' ')

            f = os.path.join(path_train, f'{strI[1]}.flac')
            attack_type = strI[3]
            label = 0 if strI[4] == 'bonafide' else 1
            if label == 0:
                train_num_neg += 1
            else:
                train_num_pos += 1
                
            item = DF_Item(f, label, attack_type, is_fake=(label == 1))
            self.train_set.append(item)

        self.class_weight.append((train_num_neg + train_num_pos) / train_num_neg)
        self.class_weight.append((train_num_neg + train_num_pos) / train_num_pos)
        
        # test_set
        test_num_pos = 0
        test_num_neg = 0
        trl = os.path.join(path_test_trl)
        for line in open(trl).readlines():
            strI = line.replace('\n', '').split(' ')
            f = os.path.join(path_test, f'{strI[1]}.flac')
            attack_type = strI[4]
            label = 0 if attack_type == '-' else 1
            if label == 0:
                test_num_neg += 1
            else:
                test_num_pos += 1
                
            item = DF_Item(f, label, attack_type, is_fake=label == 1)
            self.test_set.append(item)

        # error check
        assert len(self.test_set) == self.NUM_TEST_ITEM, f'[DATASET ERROR] - TEST_SAMPLE: {len(self.test_set)}, EXPECTED: {self.NUM_TEST_ITEM}'