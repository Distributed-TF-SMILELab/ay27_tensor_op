# Created by ay27 at 16/12/4
import numpy as np

from src import validator
from src.checker import type_check
from src.tensor import Tensor
import src.mat_utils as utils


def HOSVD(tensor):
    order = len(tensor.shape)
    Us = []
    for ii in range(order):
        X = tensor.t2mat(ii, list(set(list(range(order))) - {ii})).eval()
        U, s, V = np.linalg.svd(X)
        # U = U[:, 0:10]
        Us.append(U)

    g = tensor.ttm(Us, -1, transpose=True)
    xx = g.ttm(Us, -1)
    return g, Us, validator.RMSE(tensor, xx)


def HOOI(tensor, R=10, iters=20, tol=1e-10):
    g, Us, _ = HOSVD(tensor)
    order = len(tensor.shape)
    for iter in range(iters):
        for ii in range(order):
            if ii == 0:
                y = tensor.ttm(Us[1].T, 1)
                start = 2
            else:
                y = tensor.ttm(Us[0].T, 0)
                start = 1
            for k in range(start, order):
                if k == ii:
                    continue
                y = y.ttm(Us[k].T, k)
            U, _, _ = np.linalg.svd(y.t2mat(ii, list(set(list(range(order))) - {ii})).eval())
            Us[ii] = U[:, 0:R]

        g = tensor.ttm(Us, -1, True)
        xx = g.ttm(Us, -1)
        print('iter %d, rmse=%f' % (iter, validator.RMSE(xx, tensor)))
    return g, validator.RMSE(xx, tensor)
