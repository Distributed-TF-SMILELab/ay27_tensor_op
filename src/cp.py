# Created by ay27 at 16/12/5
import numpy as np
import src.mat_utils as utils
from src import validator
from src.tensor import Tensor


def cp(tensor, R=20, iters=200):
    X1 = tensor.t2mat(0, [1, 2]).eval()
    X2 = tensor.t2mat(1, [0, 2]).eval()
    X3 = tensor.t2mat(2, [1, 0]).eval()

    B = np.random.rand(tensor.shape[1], R)
    C = np.random.rand(tensor.shape[2], R)

    for iter in range(iters):
        t1 = utils.khatriRao(C, B)
        t2 = np.matmul(C.T, C) * np.matmul(B.T, B)
        A = X1.dot(t1).dot(np.linalg.pinv(t2))

        t1 = utils.khatriRao(A, C)
        t2 = np.matmul(A.T, A) * np.matmul(C.T, C)
        B = X2.dot(t1).dot(np.linalg.pinv(t2))

        t1 = utils.khatriRao(B, A)
        t2 = np.matmul(B.T, B) * np.matmul(A.T, A)
        C = X3.dot(t1).dot(np.linalg.pinv(t2))

        print('iter %d, rmse = %f' % (iter, validator.RMSE(X1, A.dot(utils.khatriRao(C, B).T))))


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

            if ii == 0:
                tmp = As[order - 1]
                start = order - 2
            else:
                tmp = As[ii - 1]
                start = ii - 2
                for k in range(start, -1, -1):
                    tmp = utils.khatriRao(tmp, As[k])
                start = order - 1
            for k in range(start, ii, -1):
                tmp = utils.khatriRao(tmp, As[k])

            Anew = np.matmul(np.matmul(X[ii], tmp), np.linalg.pinv(V))
            if iter == 0:
                lamd = np.linalg.norm(Anew, axis=0)
            else:
                lamd = np.linalg.norm(Anew, ord=np.inf, axis=0)
            for k in range(R):
                Anew[:, k] = Anew[:, k] / lamd[k]
            As[ii] = Anew

        tmp = As[order - 1]
        for k in range(order - 2, 0, -1):
            tmp = utils.khatriRao(tmp, As[k])
        sig = np.zeros(As[0].shape)
        for k in range(R):
            sig[:, k] = As[0][:, k] * lamd[k]
        xx = np.matmul(sig, tmp.T)
        xx = Tensor(xx.reshape(tensor.shape))
        rmse = validator.RMSE(xx, tensor)
        print('cp_als iter %d, rmse = %f' % (iter, rmse))

    return As, lamd, rmse
