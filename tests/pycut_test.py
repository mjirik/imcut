#! /usr/bin/python
# -*- coding: utf-8 -*-



# import funkcí z jiného adresáře
import sys
import os.path
import copy

path_to_script = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(path_to_script, "../src/"))
import unittest

import numpy as np

import pycut



class PycutTest(unittest.TestCase):

    # @TODO znovu zprovoznit test
    #@unittest.skip("Cekame, az to Tomas opravi")

    def test_ordered_values_by_indexes(self):
        """
        test of pycut.__ordered_values_by_indexes
        """
        slab = {'none':0, 'liver':1, 'porta':2, 'lesions':6}
        voxelsize_mm = np.array([1.0,1.0,1.2])

        # there must be some data
        img = np.zeros([32,32,32], dtype=np.int16)
        data = np.array([
            [0, 1, 1],
            [0, 2, 2],
            [0, 2, 2]
        ])

        inds = np.array([
            [0, 1, 2],
            [3, 4, 4],
            [5, 4, 4]
        ])
        gc = pycut.ImageGraphCut(img)
        vals = gc._ImageGraphCut__ordered_values_by_indexes(data, inds)
        expected = np.array([0, 1, 1, 0, 2, 0])
        self.assertItemsEqual(vals, expected)



if __name__ == "__main__":
    unittest.main()
