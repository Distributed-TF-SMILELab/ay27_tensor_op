# Created by ay27 at 16/12/4
from _operator import add
from functools import reduce

import math

from src.checker import type_check
from src.tensor import Tensor
import numpy as np


def RMSE(A, B):
    if isinstance(A, Tensor):
        a = A.vectorization()
    else:
        A = np.asanyarray(A)
        a = np.reshape(A, -1)

    if isinstance(B, Tensor):
        b = B.vectorization()
    else:
        B = np.asanyarray(B)
        b = np.reshape(B, -1)

    if A.shape != B.shape:
        raise ValueError('the shape of A and B must be equal')

    t = reduce(add, np.power(a - b, 2))

    return math.sqrt(t / len(a))
