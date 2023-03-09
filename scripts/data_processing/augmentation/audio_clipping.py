import random
import numpy as np

def audio_clipping(x, ratio):
    ratio = random.uniform(ratio[0], ratio[1])
    threshold_min = np.min(x) * ratio
    threshold_max = np.max(x) * ratio
    x = np.clip(x, a_min=threshold_min, a_max=threshold_max)
    return x
