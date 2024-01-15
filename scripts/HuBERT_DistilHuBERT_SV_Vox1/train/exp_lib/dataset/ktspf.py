import os

from ._dataclass import DF_Item

class KTSPFv1:
    NUM_VOCAB = 32
    
    def __init__(self, path):
        self.train_set = []
        
        # train_set
        for root, _, files in os.walk(path):
            for file in files:
                if '.wav' in file:
                    self.train_set.append(
                        DF_Item(
                            os.path.join(root, file),
                            label=1, 
                            attack_type=root, 
                            is_fake=True)
                    )