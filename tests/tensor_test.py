# Created by ay27 at 16/11/8
import unittest
import numpy as np
from src.tensor import Tensor


class MyTestCase(np.testing.TestCase):
    def test_t2mat(self):
        # shape of tmp 3x4x2
        tmp = [[[1, 13], [4, 16], [7, 19], [10, 22]],
               [[2, 14], [5, 17], [8, 20], [11, 23]],
               [[3, 15], [6, 18], [8, 21], [12, 24]]]
        X = np.array(tmp)
        t = Tensor(X)

        # mode-1
        Y = np.transpose(X, [0, 2, 1])
        Z = Y.reshape([3, 8])
        np.testing.assert_array_equal(Z, t.t2mat(0, [2, 1]))

        # mode-2
        Y = np.transpose(X, [1, 2, 0])
        Z = Y.reshape([4, 6])
        np.testing.assert_array_equal(Z, t.t2mat(1, [2, 0]))

        # mode-3
        Y = np.transpose(X, [2, 1, 0])
        Z = Y.reshape([2, 12])
        np.testing.assert_array_equal(Z, t.t2mat(2, [1, 0]))


if __name__ == '__main__':
    unittest.main()
