import random
import numpy as np
import soundfile as sf

def rand_crop(wav, size):
    if wav.shape[0] < size:
        wav = _pad_wav(wav, size)
    else:
        start = random.randint(0, wav.shape[0] - size)
        wav = wav[start:start + size]
    return wav
    
def rand_crop_read(path: str, size: int, dtype='float32'):
    '''Read '~.wav' file and crop it. 
    If wav is short, wrap it. 
    '''
    sf_info = sf.info(path)
    num_frame = int((sf_info.duration - 0.02) * sf_info.samplerate)  
    
    if size < num_frame:
        start = random.randint(0, num_frame - size)
        wav, _ = sf.read(path, start=start, stop=start + size, dtype=dtype)
    else:
        wav, _ = sf.read(path, dtype=dtype)
        if wav.shape[0] < size:
            wav = _pad_wav(wav, size)
        elif size < wav.shape[0]:
            wav = wav[:size]
    return wav
                    
def linspace_crop_read(path: str, num_seg: int, seg_size: int, get_org=False, dtype='float32'):
    '''Read '~.wav' file and divide it into several segments using linspace function.
    '''
    wav, _ = sf.read(path, dtype=dtype)
    
    buffer = []
    indices = np.linspace(0, wav.shape[0] - seg_size, num_seg)
    for idx in indices:
        idx = int(idx)
        buffer.append(wav[idx:idx + seg_size])
    buffer = np.stack(buffer, axis=0)
    
    if get_org:
        return buffer, wav
    else:
        return buffer

def _pad_wav(wav, size):
    if len(np.shape(wav)) == 1:
        shortage = size - wav.shape[0]
        wav = np.pad(wav, (0, shortage), 'wrap')
        return wav
    elif len(np.shape(wav)) == 2:
        shortage = size - wav.shape[0]
        wav = np.pad(wav, ((0, shortage), (0, 0)), 'wrap')
        return wav
    else:
        raise Exception()