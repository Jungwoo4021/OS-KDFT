import math
import random
import numpy as np

class EasingAugmentation:
    def __init__(self):
        self.size = 0
        self.windows = {}

    def __call__(self, x):
        if self.size != len(x):
            self.update_window(len(x))
        
        key = random.sample(self.windows.keys(), 1)[0]
        window = self.windows[key].copy()

        if random.random() < 0.5:
            window = np.flip(window)

        b = random.uniform(0.1, 0.3)

        window = window * (1 - b)

        x = x * (b + window)
        
        return x

    def update_window(self, size):
        self.size = size
        self.windows = {}

        self.windows['sine'] = np.linspace(0, 1, size)
        self.windows['quad'] = np.linspace(0, 1, size)
        self.windows['cubic'] = np.linspace(0, 1, size)
        self.windows['circ'] = np.linspace(0, 1, size)
        self.windows['expo'] = np.linspace(0, 1, size)
        self.windows['bounce'] = np.linspace(0, 1, size)

        for i in range(size):
            self.windows['sine'][i] = self._sin(self.windows['sine'][i])
            self.windows['quad'][i] = self._quad(self.windows['quad'][i])
            self.windows['cubic'][i] = self._cubic(self.windows['cubic'][i])
            self.windows['circ'][i] = self._circ(self.windows['circ'][i])
            self.windows['expo'][i] = self._expo(self.windows['expo'][i])
            self.windows['bounce'][i] = self._bounce(self.windows['bounce'][i])

    def _sin(self, x):
        return 1 - math.cos((x * math.pi) * 0.5)

    def _quad(self, x):
        return x * x

    def _cubic(self, x):
        return x * x * x

    def _circ(self, x):
        return 1 - np.sqrt(1 - (x ** 2))

    def _expo(self, x):
        return 0 if x == 0 else 2 ** (10 * x - 10)

    def _bounce(self, x):
        n1 = 7.5625
        d1 = 2.75

        if (x < 1 / d1):
            return n1 * x * x
        elif (x < 2 / d1):
            t = x - 1.5 / d1
            return n1 * t * x + 0.75
        elif (x < 2.5 / d1):
            t = x - 2.25 / d1
            return n1 * t * x + 0.9375
        else:
            t = x - 2.625 / d1
            return n1 * t * x + 0.984375