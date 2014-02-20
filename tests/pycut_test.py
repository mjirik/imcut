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

        segm = np.zeros([256,256,80], dtype=np.int16)

        # liver
        segm[70:190,40:220,30:60] = slab['liver']
# port
        segm[120:130,70:220,40:45] = slab['porta']
        segm[80:130,100:110,40:45] = slab['porta']
        segm[120:170,130:135,40:44] = slab['porta']

        # vytvoření kopie segmentace - před určením lézí

if __name__ == "__main__":
    unittest.main()
