# Created by ay27 at 16/12/4
import unittest
import src.mat_utils as utils
import numpy as np
import time


class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.A = np.random.randn(40, 30)
        self.B = np.random.randn(20, 30)

    def test_kroncker(self):
        res = utils.kron(self.A, self.B)
        tmp = np.zeros((self.A.shape[0] * self.B.shape[0], self.A.shape[1] * self.B.shape[1]))
        for ii in range(self.A.shape[0]):
            for jj in range(self.A.shape[1]):
                for i in range(self.B.shape[0]):
                    for j in range(self.B.shape[1]):
                        tmp[ii * self.B.shape[0] + i, jj * self.B.shape[1] + j] = self.A[ii, jj] * self.B[i, j]
        np.testing.assert_array_almost_equal(res, tmp)

    def test_khatriRao(self):
        res = utils.khatriRao(self.A, self.B)
        tmp = np.zeros((self.A.shape[0] * self.B.shape[0], self.A.shape[1]))
        for kk in range(self.A.shape[1]):
            for ii in range(self.A.shape[0]):
                for jj in range(self.B.shape[0]):
                    tmp[ii * self.B.shape[0] + jj, kk] = self.A[ii, kk] * self.B[jj, kk]
        np.testing.assert_array_almost_equal(res, tmp)

    def test_err(self):
        with self.assertRaises(ValueError):
            utils.khatriRao(np.random.randn(2, 3), np.random.randn(3, 4))


if __name__ == '__main__':
    unittest.main()
