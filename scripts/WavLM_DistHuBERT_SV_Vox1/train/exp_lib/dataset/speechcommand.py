import os
from ._dataclass import KS_Item

class SpeechCommand_v001:
    def __init__(self, path):
        self.label_dict = {}
        self.train_set = self.parse_txt(os.path.join(path, 'train_list.txt'), path)
        self.val_set = self.parse_txt(os.path.join(path, 'validation_list.txt'), path)
        self.eval_set = self.parse_txt(os.path.join(path, 'testing_list.txt'), path)
        
    def parse_txt(self, path, path_data):
        items = []
        for line in open(path).readlines():
            line = line.replace('\n', '')
            strI = line.split('/')
            try: self.label_dict[strI[0]]
            except: 
                self.label_dict[strI[0]] = len(self.label_dict.keys())
            l = self.label_dict[strI[0]]
            f = os.path.join(path_data, strI[0], strI[1])
            items.append(
                KS_Item(
                    path=f, label=l
                )
            )
        return items