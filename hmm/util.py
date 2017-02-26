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

def log_vec_mul(*vec_arrays):
    return np.sum(vec_arrays, axis=0)

def log_vec_mul_c(array, c):
    return arrays + c

def log_div(*arrays):
    return arrays[0] - log_mul(*arrays[1:])

def log_vec_div(*vec_arrays):
    return vec_arrays[0] - log_vec_mul(*vec_arrays[1:])

def log_vec_div_c(array, c):
    return array - c

def log_add(*arrays):
    a = max(arrays[0], arrays[1])
    b = min(arrays[0], arrays[1])
    return a + math.log(1.0 + math.exp(b-a))

def log_sum(*arrays):
    return reduce(lambda x,y: log_add(x, y), arrays)
