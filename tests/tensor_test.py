# Created by ay27 at 16/11/8
import unittest
from functools import reduce

import numpy as np
import tensorflow as tf
from src.tensor import Tensor


class TensorTest(np.testing.TestCase):
    def setUp(self):
        # shape of tmp 3x4x2
        self.tmp = [[[1, 13], [4, 16], [7, 19], [10, 22]],
                    [[2, 14], [5, 17], [8, 20], [11, 23]],
                    [[3, 15], [6, 18], [9, 21], [12, 24]]]
        self.X = np.array(self.tmp)
        self.t = Tensor(self.X)

    def test_value(self):
        with self.assertRaises(ValueError):
            t = Tensor((1, 2, 3))
        t = Tensor([_ for _ in range(100)])

    def test_t2mat(self):

        s = tf.Session()
        with s.as_default():
            # mode-1
            Y = np.transpose(self.X, [0, 2, 1])
            Z = Y.reshape([3, 8])
            np.testing.assert_array_equal(Z, self.t.t2mat(0, [2, 1]).eval())

            # mode-2
            Y = np.transpose(self.X, [1, 2, 0])
            Z = Y.reshape([4, 6])
            np.testing.assert_array_equal(Z, self.t.t2mat(1, [2, 0]).eval())

            # mode-3
            Y = np.transpose(self.X, [2, 1, 0])
            Z = Y.reshape([2, 12])
            np.testing.assert_array_equal(Z, self.t.t2mat(2, [1, 0]).eval())

    def test_err(self):
        try:
            self.t.t2mat([2, 3], [4, 5]).eval()
        except:
            self.assertTrue(True)

        try:
            self.t.t2mat(3, [0, 1, 2]).eval()
        except:
            self.assertTrue(True)

    def test_reverse(self):
        # 2x4
        U = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])

        # mode-2
        # X is a 3x4x2_p tensor, after transpose, X is a 4x2x3 tensor; then reshape it
        Z = np.transpose(self.X, [1, 2, 0]).reshape([4, 6])  # 4x 2_p x3 => 4x6
        Z = U.dot(Z)  # 2x6
        # Z is 2x6 matrix, reshape it back to 2_f x 2_p x3; transpose it into 3x 2_f x2_p
        reverse1 = np.transpose(Z.reshape([2, 2, 3]), [2, 0, 1])  # 2x6 => 2_1x3x2_2 => 3x2x2

        Z_ = np.transpose(self.X, [1, 0, 2]).reshape([4, 6])  # 4x3x2 => 4x6
        Z_ = U.dot(Z_)  # 2x6
        reverse2 = np.transpose(Z_.reshape([2, 3, 2]), [1, 0, 2])  # 2x6 => 2x3x2 => 3x2x2

        np.testing.assert_array_equal(reverse1, reverse2)

    def test_norm(self):
        x = reduce(lambda x, y: x + y, np.power(self.X.reshape(-1), 2))
        self.assertEqual(pow(self.t.norm(), 2), x)

        x = reduce(lambda x, y: x + y, np.power(self.X.reshape(-1), 4))
        self.assertAlmostEqual(pow(self.t.norm(4), 4), x)

    def test_inner(self):
        pass

if __name__ == '__main__':
    unittest.main()
