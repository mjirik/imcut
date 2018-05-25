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

orig_sr_tab = {
    2: np.array([(0,2), (0,1), (1,3), (2,3)]),
    3: np.array([(0,3,6), (0,1,2), (2,5,8), (6,7,8)]),
    4: np.array([(0,4,8,12), (0,1,2,3), (3,7,11,15), (12,13,14,15)]),
}

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
        g

    unittest.skip("waiting for fix")
    def test_graph_3d_two_slices(self):

        data = np.array(
            [
                [[0,0,0,0,1,1,0],
                 [0,1,1,0,1,0,1],
                 [0,1,1,0,1,0,0],
                 [1,1,1,0,0,0,0],
                 [0,0,0,0,0,0,1]],
                [[0,0,0,0,0,0,0],
                 [0,0,0,0,0,0,0],
                 [0,1,1,0,0,0,0],
                 [0,1,1,0,0,0,0],
                 [0,0,0,0,0,0,1]],
            ]
        )
        g = graph.Graph(data, (0.1, 0.12, 0.05))
        g

    unittest.skip("waiting for fix")
    def test_graph_3d(self):

        data = np.array(
            [
                [[0,0,0,0,0],
                 [0,1,1,0,1],
                 [1,1,1,0,0],
                 [0,0,0,0,1]],
                [[0,0,0,0,0],
                 [0,0,0,0,0],
                 [0,1,1,0,0],
                 [0,0,0,0,1]],
                [[0,0,0,0,0],
                 [0,0,0,0,0],
                 [0,1,1,0,0],
                 [0,0,0,0,1]],
            ]
        )
        g = graph.Graph(data, (0.1, 0.12, 0.05))
        g

    def _test_automatic_ms_indexes_2d_same_as_orig(self, size):
        shape = [size, size]
        srt = graph.SRTab(shape)
        subtab = srt.get_sr_subtab()

        err = np.sum(np.abs(subtab - orig_sr_tab[size]))
        self.assertEqual(err, 0)

    def test_automatic_ms_indexes_2d_same_as_orig_2(self):
        size = 2
        self._test_automatic_ms_indexes_2d_same_as_orig(size)

    def test_automatic_ms_indexes_2d_same_as_orig_3(self):
        size = 3
        self._test_automatic_ms_indexes_2d_same_as_orig(size)

    def test_automatic_ms_indexes_2d_same_as_orig_4(self):
        size = 4
        self._test_automatic_ms_indexes_2d_same_as_orig(size)


    def test_automatic_ms_indexes_3d(self):
        shape = [3, 3, 3]
        srt = graph.SRTab(shape)
        subtab = srt.get_sr_subtab()
        subtab

        # err = np.sum(np.abs(subtab - orig_sr_tab[size]))
        # self.assertEqual(err, 0)

if __name__ == "__main__":
    unittest.main()
