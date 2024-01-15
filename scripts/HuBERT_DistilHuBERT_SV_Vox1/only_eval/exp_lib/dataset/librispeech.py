import os

from ._dataclass import ASR_Item

class Libri_100h:
    NUM_VOCAB = 32
    
    def __init__(self, path):
        self.train_set = []
        
        # train_set
        for root, _, files in os.walk(path):
            for file in files:
                if '.trans.txt' in file:
                    f = open(os.path.join(root, file))
                    for line in f.readlines():
                        line = line.replace('\n', '')
                        strI = line.split(' ')
                        self.train_set.append(
                            ASR_Item(os.path.join(root, strI[0] + '.wav'), ' '.join(strI[1:]))
                        )