import math
import numpy as np

EPS = 1e-4
ZERO = 1e-12
LOG_ZERO = math.log(1e-12)

def normalize2d(vec):
    return (vec.transpose() / vec.sum(axis=1)).transpose()

def normalize1d(vec):
    return vec / vec.sum()

class LogNum:
    def __init__(self, val=-1):
        self.val = math.log(val if val > ZERO else ZERO) if val >= 0 else -1

    def __float__(self):
        return math.exp(self.val)

    def __str__(self):
        return str(self.__float__())

    def __mul__(self, other):
        tmp = LogNum()
        tmp.val = self.val + other.val
        return tmp

    def __truediv__(self, other):
        tmp = LogNum()
        tmp.val = self.val - other.val
        return tmp

    def __add__(self, other):
        tmp = LogNum()
        a = max(self.val, other.val)
        b = min(self.val, other.val)
        tmp.val = a + math.log(1.0 + math.exp(b-a))
        return tmp

    def __lt__(self, other):
        return self.val < other.val

    def __gt__(self, other):
        return self.val > other.val;
