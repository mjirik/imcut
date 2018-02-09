#! /usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function

# import funkcí z jiného adresáře
import sys
import os.path
import unittest
import scipy
import numpy as np

import logging
logger = logging.getLogger(__name__)

path_to_script = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(path_to_script, "../src/"))

from nose.plugins.attrib import attr
from pysegbase import models


class ModelsTest(unittest.TestCase):
    # @classmethod
    # def setUpClass(cls):
    #     if sys.version_info.major < 3:
    #         cls.assertCountEqual = cls.assertItemsEqual

    def test_sigmoid(self):
        # from pysegbase import models
        import matplotlib.pyplot as plt
        x = np.linspace(-1000,1000, 100)
        min = 20
        max = 100
        multiplicator = max - min
        val = multiplicator * models.sigmoid(min + x * 1.0/multiplicator)
        # plt.plot(x, val)
        # plt.show()
        self.assertTrue(np.all(val >= 0))

    def test_softplus(self):
        import matplotlib.pyplot as plt
        x = np.linspace(-1000,1000, 1000)
        maximum_error = 10
        val = models.softplus(x, maximum_error)
        # plt.plot(x, val)
        # plt.show()
        self.assertTrue(np.all(val >= 0))
        self.assertTrue(np.all(val[x < 0] < maximum_error))
        # print(val)

if __name__ == "__main__":
    unittest.main()
