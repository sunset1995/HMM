import math
import numpy as np
from functools import reduce

EPS = 1e-4
ZERO = 1e-300
LOG_ZERO = math.log(ZERO)

def normalize2d(vec):
    return (vec.transpose() / vec.sum(axis=1)).transpose()

def normalize1d(vec):
    return vec / vec.sum()

def log_mul(*arrays):
    return np.array(arrays).sum(axis=0)

log_vec_mul = np.vectorize(log_mul)

def log_div(*arrays):
    return arrays[0] - log_mul(*arrays[1:])

log_vec_div = np.vectorize(log_div)

def log_add(x, y):
    a = max(x, y)
    b = min(x, y)
    return a + math.log(1.0 + math.exp(b-a))

log_vec_add = np.vectorize(log_add)

def log_sum(*arrays):
    return reduce(lambda x,y: log_add(x, y), arrays)
