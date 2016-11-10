# Created by ay27 at 16/11/8
import numpy as np
import tensorflow as tf
from src.arg_check import arg_check


class Tensor:
    @arg_check(None, [np.ndarray, np.matrix, list])
    def __init__(self, data):
        self.data = np.array(data)
        self.shape = self.data.shape

    def vectorization(self):
        return self.data.reshape(-1)

    @arg_check(None, [list, int], [list, int])
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
            indies.extend(cdims)
            csize = np.prod([self.shape[i] for i in cdims])
        tmp = tf.reshape(tf.transpose(self.data, indies), (rsize, csize))
        return tmp

    def __cmp__(self, other):
        t1 = self.data.reshape(-1)
        t2 = other.data.reshape(-1)
        if len(t1) != len(t2):
            return False
        for ii in range(len(t1)):
            if t1[ii] != t2[ii]:
                return False
        return True
