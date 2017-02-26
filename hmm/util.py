import math
import numpy as np
from functools import reduce

EPS = 1e-4
ZERO = 1e-12
LOG_ZERO = math.log(1e-12)

def normalize2d(vec):
    return (vec.transpose() / vec.sum(axis=1)).transpose()

def normalize1d(vec):
    return vec / vec.sum()

def log_mul(*arrays):
    print(arrays)
    return np.array(arrays).sum(axis=0)

def log_div(*arrays):
    return arrays[0] - log_mul(*arrays[1:])

def log_add(*arrays):
    a = max(arrays[0], arrays[1])
    b = min(arrays[0], arrays[1])
    return a + math.log(1.0 + math.exp(b-a))

def log_sum(*arrays):
    return reduce(lambda x,y: log_add(x, y), arrays)
