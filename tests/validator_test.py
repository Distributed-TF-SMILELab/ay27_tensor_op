# Created by ay27 at 16/12/4
import unittest

import math
import numpy as np

from src.tensor import Tensor
from src.validator import RMSE


class MyTestCase(unittest.TestCase):
    def test_rmse(self):
        a = Tensor(np.random.randn(30, 20))
        b = Tensor(np.random.randn(30, 20))
        ans = RMSE(a, b)
        tmp = 0.0
        for ii in range(a.shape[0]):
            for jj in range(a.shape[1]):
                tmp += (a.data[ii, jj] - b.data[ii, jj]) * (a.data[ii, jj] - b.data[ii, jj])
        tmp /= 600.0
        tmp = math.sqrt(tmp)
        self.assertAlmostEqual(tmp, ans)


if __name__ == '__main__':
    unittest.main()
