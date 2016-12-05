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

        try:
            self.t.t2mat('sad', 123).eval()
        except ValueError:
            self.assertTrue(True)

    def test_ttm(self):
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

        s = tf.Session()
        with s.as_default():
            result = self.t.ttm(U, 1)
            np.testing.assert_array_equal(reverse2, result.data)

        np.testing.assert_array_equal(reverse1, reverse2)

        U = np.array([1, 2, 3, 4])
        ground_true = [[70, 190], [80, 200], [90, 210]]
        with s.as_default():
            result = self.t.ttv(U, 1)
            np.testing.assert_array_equal(result.data, ground_true)

            result = self.t.ttm(U, 1)
            np.testing.assert_array_almost_equal(result.data.reshape(-1), np.array(ground_true).reshape(-1))

        U = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
        with s.as_default():
            result = self.t.ttm(U, 0)
            tmp = [[[14, 86], [32, 104], [50, 122], [68, 140]],
                   [[32, 212], [77, 257], [122, 302], [167, 347]],
                   [[50, 338], [122, 410], [194, 482], [266, 554]],
                   [[68, 464], [167, 563], [266, 662], [365, 761]]]
            np.testing.assert_array_almost_equal(result.data, tmp)

            U = np.random.rand(3,2)
            result = self.t.ttm(U.T, 0)
            print(result)

    def test_ttt(self):
        # t: 3x4x2, t1: 4x2x3
        with tf.Session().as_default():
            t1 = Tensor(self.t.data.transpose([1, 2, 0]))
            tmp = t1.ttt(self.t, [2, 0], [0, 1])
            ans = [[650, 1586], [1586, 4250]]
            np.testing.assert_array_almost_equal(tmp.data, ans)

            tmp = t1.ttt(self.t, [2, 0, 1], [0, 1, 2])
            ans = [4900]
            np.testing.assert_array_almost_equal(tmp.data, ans)

            ans = [[1436, 1528, 1620],
                   [1528, 1628, 1728],
                   [1620, 1728, 1836]]
            tmp = t1.ttt(self.t, [0, 1], [1, 2])
            np.testing.assert_array_almost_equal(tmp.data, ans)

    def test_norm(self):
        x = reduce(lambda x, y: x + y, np.power(self.X.reshape(-1), 2))
        self.assertEqual(pow(self.t.norm(), 2), x)

        x = reduce(lambda x, y: x + y, np.power(self.X.reshape(-1), 4))
        self.assertAlmostEqual(pow(self.t.norm(4), 4), x)

    def test_inner(self):
        tmp = np.random.randn(3, 4, 2)
        r1 = tmp.reshape(-1).dot(self.X.reshape(-1).T)

        r2 = self.t.inner(Tensor(tmp))

        self.assertAlmostEqual(r1, r2)

    def test_cmp(self):
        tmp = Tensor(np.random.randn(3, 4, 2))
        self.assertNotEqual(self.t, tmp)
        tmp = Tensor(self.tmp)
        self.assertEqual(self.t, tmp)

    def test_print(self):
        groud = '[:,:,0]\n' \
                '[[ 1  4  7 10]\n' \
                ' [ 2  5  8 11]\n' \
                ' [ 3  6  9 12]]\n' \
                '[:,:,1]\n' \
                '[[13 16 19 22]\n' \
                ' [14 17 20 23]\n' \
                ' [15 18 21 24]]\n'
        result = self.t.__repr__()
        self.assertEqual(groud, result)


if __name__ == '__main__':
    unittest.main()
