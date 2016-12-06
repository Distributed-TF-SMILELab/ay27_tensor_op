# Created by ay27 at 16/12/4
import numpy as np

from src import validator
from src.checker import type_check
from src.tensor import Tensor


@type_check(Tensor)
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


@type_check(Tensor)
def HOOI(tensor, R=10, iters=20, tol=1e-2):
    g, Us, _ = HOSVD(tensor)
    order = len(tensor.shape)
    rmse = 0.0
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
        rmse_n = validator.RMSE(xx, tensor)
        print('iter %d, rmse=%f' % (iter, rmse_n))
        if abs(abs(rmse_n - rmse) - tol) < tol:
            return g, rmse_n
        rmse = rmse_n
    return g, rmse_n
