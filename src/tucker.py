# Created by ay27 at 16/12/4
import numpy as np

from src import validator
from src.checker import type_check
from src.tensor import Tensor
import src.mat_utils as utils


@type_check(Tensor, int)
def tucker_als(tensor, iters=20, tol=1e-10):
    X1 = tensor.t2mat(0, [1, 2]).eval()
    X2 = tensor.t2mat(1, [2, 0]).eval()
    X3 = tensor.t2mat(2, [0, 1]).eval()

    B = np.random.randn(tensor.shape[1], 20)
    C = np.random.randn(tensor.shape[2], 20)

    for iter in range(iters):
        A, _, _ = np.linalg.svd(np.matmul(X1, utils.kron(C, B)), 0)
        B, _, _ = np.linalg.svd(np.matmul(X2, utils.kron(A, C)), 0)
        C, _, _ = np.linalg.svd(np.matmul(X3, utils.kron(B, A)), 0)
        A = A[:, 0:2]
        B = B[:, 0:2]
        C = C[:, 0:2]

        # G3 = np.matmul(np.matmul(C.T, X3), utils.kron(B, A))
        # G2 = np.matmul(np.matmul(B.T, X2), utils.kron(A, C))
        # G1 = np.matmul(np.matmul(A.T, X1), utils.kron(C, B))
        G = tensor.ttm(A.T, 0).ttm(B.T, 1).ttm(C.T, 2)
        xx = G.ttm(A, 0).ttm(B, 1).ttm(C, 2)
        # print(xx)
        rmse = validator.RMSE(tensor, xx)
        print('iter %d : rmse = %f' % (iter, rmse))
        if abs(rmse - tol) < tol:
            break
    return G, A, B, C, rmse


def HOSVD(tensor):
    order = len(tensor.shape)
    Us = []
    for ii in range(order):
        X = tensor.t2mat(ii, list(set(list(range(order))) - {ii})).eval()
        U, s, V = np.linalg.svd(X)
        U = U[:, 0:10]
        Us.append(U)

    g = tensor.ttm(Us[0].T, 0)
    for ii in range(1, order):
        g = g.ttm(Us[ii].T, ii)

    xx = g.ttm(Us[0], 0)
    for ii in range(1, order):
        xx = xx.ttm(Us[ii], ii)
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
    g = tensor.ttm(Us[0].T, 0)
    for ii in range(1, order):
        g = g.ttm(Us[ii].T, ii)

    xx = g.ttm(Us[0], 0)
    for ii in range(1, order):
        xx = xx.ttm(Us[ii], ii)

    return g, validator.RMSE(xx, tensor)
