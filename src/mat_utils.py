# Created by ay27 at 16/12/4
import numpy as np
from src.checker import type_check


def kron(A, B):
    A = np.asarray(A)
    B = np.asarray(B)
    return np.kron(A, B)


@type_check([np.ndarray, np.array, np.mat, list], [np.ndarray, np.array, np.mat, list])
def khatriRao(A, B):
    if isinstance(A, list):
        A = np.array(A)
    if isinstance(B, list):
        B = np.array(B)
    if not (len(A.shape) == len(B.shape) == 2):
        raise ValueError('A and B must be a matrix')
    if A.shape[1] != B.shape[1]:
        raise ValueError('the column of A and B must be equaled')
    tmp = []
    AT = A.T
    BT = B.T
    for k in range(A.shape[1]):
        tmp.append(np.kron(AT[k], BT[k]))
    return np.array(tmp).T


@type_check([np.ndarray, np.array, np.mat, list], [np.ndarray, np.array, np.mat, list])
def hadamard(A, B):
    if isinstance(A, list):
        A = np.array(A)
    if isinstance(B, list):
        B = np.array(B)
    return A * B
