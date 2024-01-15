from dataclasses import dataclass

@dataclass
class SV_TrainItem:
    path    : str
    speaker : str
    label   : int

@dataclass
class SV_EnrollmentItem:
    key     : str
    path    : str
    speaker : str = None

@dataclass
class SV_Trial:
    key1    : str
    key2    : str
    label   : int

@dataclass
class DF_Item:
    path        : str
    label       : int
    attack_type : str
    is_fake     : bool

@dataclass
class MusicGenreClassificationData:
    path: str
    genre: str
    label: int

@dataclass
class SID_TrainItem:
    path    : str
    speaker : str
    label   : int

@dataclass
class SID_TestItem:
    path    : str
    speaker : str
    label   : int
    
@dataclass
class KS_Item:
    path    : str
    label   : int
    
@dataclass
class ASR_Item:
    path    : str
    label   : int