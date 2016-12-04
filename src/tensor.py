# Created by ay27 at 16/11/8
import numpy as np
import tensorflow as tf
from src.checker import type_check


class Tensor:
    @type_check(None, [np.ndarray, np.matrix, list])
    def __init__(self, data):
        self.data = np.copy(np.array(data))
        self.shape = self.data.shape
        self.__vec__ = None

    def vectorization(self):
        """
        the __vec__ is bound with self.data
        :return:
        """
        if self.__vec__ is None:
            self.__vec__ = self.data.reshape(-1)
        return self.__vec__

    @type_check(None, [list, int], [list, int])
    def t2mat(self, rdims, cdims):
        if isinstance(rdims, int):
            indies = [rdims]
            rsize = self.shape[rdims]
        else:
            indies = rdims
            rsize = np.prod([self.shape[i] for i in rdims])
        if isinstance(cdims, int):
            indies.append(cdims)
            csize = self.shape[cdims]
        else:
            indies = indies + cdims
            # indies.extend(cdims)
            csize = np.prod([self.shape[i] for i in cdims])
        tmp = tf.reshape(tf.transpose(self.data, indies), (int(rsize), int(csize)))

        return tmp

    def inner(self, B):
        if not isinstance(B, Tensor):
            raise ValueError('B must be a Tensor object')
        if not np.array_equal(self.shape, B.shape):
            raise ValueError('the shape of B must be equal as self')
        return np.inner(self.vectorization(), B.vectorization())

    @type_check(None, int)
    def norm(self, p=2):
        """
        :param p: {non-zero int, inf, -inf, 'fro', 'nuc'}, optional
        Order of the norm (see table under ``Notes``). inf means numpy's
        `inf` object.
        :return:
        """
        return np.linalg.norm(self.vectorization(), p)

    @type_check(None, [np.ndarray, np.matrix], int)
    def ttv(self, vec, axis=0):
        return self.ttm(np.reshape(vec, [1, vec.shape[0]]), axis)

    @type_check(None, [np.ndarray, np.matrix, list], [int, list])
    def ttm(self, U, axis=0, transpose=False):
        if axis == -1:
            axis = list(range(len(U)))
        if isinstance(U, list):
            if transpose:
                tmp = self.__ttm(U[axis[0]].T, axis[0])
                for ii in range(1, len(U)):
                    tmp = tmp.__ttm(U[axis[ii]].T, axis[ii])
            else:
                tmp = self.__ttm(U[axis[0]], axis[0])
                for ii in range(1, len(U)):
                    tmp = tmp.__ttm(U[axis[ii]], axis[ii])
            return tmp
        else:
            if transpose:
                return self.__ttm(U.T, axis)
            else:
                return self.__ttm(U, axis)

    @type_check(None, [np.ndarray, np.matrix], int)
    def __ttm(self, U, axis=0):
        if len(U.shape) == 1:
            return self.ttv(U, axis)

        indies = list(range(len(self.shape)))
        indies[0], indies[axis] = indies[axis], indies[0]

        tmp = self.t2mat(axis, indies[1:]).eval()
        tmp = np.matmul(U, tmp)
        back_shape = [self.shape[_] for _ in indies]
        back_shape[0] = tmp.shape[0]
        result = np.reshape(tmp, back_shape).transpose(indies)
        # del the 1-dim
        shape = [_ for _ in result.shape if _ > 1]
        return Tensor(result.reshape(shape))

    @type_check(None, None, [int, list, tuple], [int, list, tuple])
    def ttt(self, tensor, adims=0, bdims=0):
        if isinstance(adims, int):
            adims = [adims]
        if isinstance(bdims, int):
            bdims = [bdims]
        adims = list(adims)
        bdims = list(bdims)
        if len(adims) != len(bdims):
            raise ValueError('the len of adims and bdims must be equal')
        r1 = list(set(range(len(self.shape))) - set(adims))
        r2 = list(set(range(len(tensor.shape))) - set(bdims))
        a = self.t2mat(adims, r1).eval()
        b = tensor.t2mat(bdims, r2).eval()
        c = np.matmul(a.T, b)
        return Tensor(np.reshape(c, [self.shape[_] for _ in r1] + [tensor.shape[_] for _ in r2]))

    def __eq__(self, other):
        return np.array_equal(self.data, other.data)

    def __repr__(self):
        if len(self.shape) < 3:
            return str(self.data)

        import itertools
        prod = 'itertools.product('
        fmt = 'self.data[:,:'
        for ii in range(len(self.shape) - 2):
            prod += 'range(%d),' % self.shape[ii + 2]
            fmt += ',%d'
        fmt += ']'
        prod = prod[0:-1] + ')'

        result = ''
        for t in eval(prod):
            tmp = fmt % t
            result += tmp[9:] + '\n' + str(eval(tmp)) + '\n'
        return result
