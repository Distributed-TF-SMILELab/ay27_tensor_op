# Created by ay27 at 16/11/8
import unittest

if __name__ == "__main__":
    suite = unittest.TestLoader().discover('.', pattern="*_test.py")
    unittest.TextTestRunner(verbosity=2).run(suite)
