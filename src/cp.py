# Created by ay27 at 16/12/5
import numpy as np
import src.mat_utils as utils
from src import validator
from src.tensor import Tensor


def cp_als(tensor, R=20, iters=200):
    order = len(tensor.shape)

    As = [np.random.rand(tensor.shape[_], R) for _ in range(order)]

    X = []
    for ii in range(order):
        X.append(tensor.t2mat(ii, list(set(range(order)) - {ii})).eval())

    for iter in range(iters):
        for ii in range(order):
            if ii == 0:
                V = np.matmul(As[1].T, As[1])
                start = 2
            else:
                V = np.matmul(As[0].T, As[0])
                start = 1
            for k in range(start, order):
                if k == ii:
                    continue
                V = utils.hadamard(V, np.matmul(As[k].T, As[k]))

            if ii == order - 1:
                tmp = As[order - 2]
                start = order - 3
            else:
                tmp = As[order - 1]
                start = order - 2
            for k in range(start, -1, -1):
                if k == ii:
                    continue
                tmp = utils.khatriRao(tmp, As[k])
            Anew = np.matmul(np.matmul(X[ii], tmp), np.linalg.pinv(V))
            lamd = np.linalg.norm(Anew, axis=1)
            As[ii] = Anew

        tmp = As[order - 1]
        for k in range(order - 2, 0, -1):
            tmp = utils.khatriRao(tmp, As[k])
        xx = np.matmul(As[0], tmp.T)
        xx = Tensor(xx.reshape(tensor.shape))
        rmse = validator.RMSE(xx, tensor)
        print('cp_als iter %d, rmse = %f' % (iter, rmse))

    return As, lamd, rmse
