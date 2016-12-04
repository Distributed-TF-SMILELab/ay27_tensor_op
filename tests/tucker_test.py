# Created by ay27 at 16/12/4
import numpy as np
from src.tensor import Tensor
from src.tucker import tucker_als, HOSVD, HOOI
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

    def test_tucker_als(self):
        with tf.Session().as_default():
            A = Tensor(np.abs(np.random.randn(30, 40, 50) * 10))
            g, a, b, c, rmse = tucker_als(A)
            # print(g)
            # print('--------------------')
            print('als: %f' % rmse)

    def test_HOSVD(self):
        with tf.Session().as_default():
            A = Tensor(np.abs(np.random.randn(30, 40, 50) * 10))
            g, _, rmse = HOSVD(A)
            print('HOSVD: %f' % rmse)

    def test_HOOI(self):
        with tf.Session().as_default():
            A = Tensor(np.abs(np.random.randn(30, 40, 50) * 10))
            g, rmse = HOOI(A)
            print('HOOI: %f' % rmse)


if __name__ == '__main__':
    unittest.main()
