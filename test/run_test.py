import sys, os
sys.path.append(os.getcwd())

import unittest
from source.utils.path_utils import TEST_DIR

def get_allcase():
    discover = unittest.defaultTestLoader.discover(TEST_DIR, pattern="test_*.py")
    suite = unittest.TestSuite()
    suite.addTest(discover)
    return suite

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(get_allcase())