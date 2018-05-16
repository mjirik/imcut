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
from pysegbase import graph


class GraphTest(unittest.TestCase):
    # @classmethod
    # def setUpClass(cls):
    #     if sys.version_info.major < 3:
    #         cls.assertCountEqual = cls.assertItemsEqual
    def test_graph(self):

        data = np.array([[0,0,0,0,1,1,0],
                         [0,1,1,0,1,0,1],
                         [0,1,1,0,1,0,0],
                         [1,1,1,0,0,0,0],
                         [0,0,0,0,0,0,1]])
        g = graph.Graph(data, (0.1, 0.12, 0.0))


if __name__ == "__main__":
    unittest.main()
