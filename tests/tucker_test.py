# Created by ay27 at 16/12/4
import numpy as np

from src import validator
from src.tensor import Tensor
from src.tucker import HOSVD, HOOI
import unittest
import math
from src.validator import RMSE
import tensorflow as tf


class MyTestCase(unittest.TestCase):
    def setUp(self):
        # shape of tmp 3x4x2
        self.tmp = [[[1, 13], [4, 16], [7, 19], [10, 22]],
                    [[2, 14], [5, 17], [8, 20], [11, 23]],
                    [[3, 15], [6, 18], [9, 21], [12, 24]]]
        self.X = np.array(self.tmp)
        self.t = Tensor(self.X)

    def test_HOSVD(self):
        with tf.Session().as_default():
            A = Tensor(np.abs(np.random.rand(30, 40, 50) * 10))
            g, _, rmse = HOSVD(A)
            print('HOSVD: %f' % rmse)

    def test_HOOI(self):
        with tf.Session().as_default():
            G = Tensor(np.random.rand(20, 20, 20))
            A = np.random.rand(30, 20)
            B = np.random.rand(40, 20)
            C = np.random.rand(20, 20)

            tmp = G.ttm(A, 0).ttm(B, 1).ttm(C, 2)
            tmp.data /= 100.0
            print('empty %f' % validator.RMSE(tmp, Tensor(np.zeros((30, 40, 20)))))
            g, rmse = HOOI(tmp, 2)
            print('HOOI: %f' % rmse)


if __name__ == '__main__':
    unittest.main()
