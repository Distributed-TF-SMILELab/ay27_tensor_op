# Created by ay27 at 16/11/8
import unittest
import numpy as np
import tensorflow as tf
from src.tensor import Tensor


class TensorTest(np.testing.TestCase):
    def test_value(self):
        with self.assertRaises(ValueError):
            t = Tensor((1, 2, 3))
        t = Tensor([_ for _ in range(100)])

    def test_t2mat(self):
        # shape of tmp 3x4x2
        tmp = [[[1, 13], [4, 16], [7, 19], [10, 22]],
               [[2, 14], [5, 17], [8, 20], [11, 23]],
               [[3, 15], [6, 18], [9, 21], [12, 24]]]
        X = np.array(tmp)
        t = Tensor(X)

        s = tf.Session()
        with s.as_default():
            # mode-1
            Y = np.transpose(X, [0, 2, 1])
            Z = Y.reshape([3, 8])
            np.testing.assert_array_equal(Z, t.t2mat(0, [2, 1]).eval())

            # mode-2
            Y = np.transpose(X, [1, 2, 0])
            Z = Y.reshape([4, 6])
            np.testing.assert_array_equal(Z, t.t2mat(1, [2, 0]).eval())

            # mode-3
            Y = np.transpose(X, [2, 1, 0])
            Z = Y.reshape([2, 12])
            np.testing.assert_array_equal(Z, t.t2mat(2, [1, 0]).eval())

            try:
                t.t2mat([2, 3], [4, 5]).eval()
            except:
                self.assertTrue(True)

            try:
                t.t2mat(3, [0, 1, 2]).eval()
            except:
                self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
