#! /usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function

# import funkcí z jiného adresáře
import sys
import os.path
import unittest
import scipy
import numpy as np
import pysegbase.image_manipulation as imma

import logging
logger = logging.getLogger(__name__)

path_to_script = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(path_to_script, "../src/"))

from nose.plugins.attrib import attr
from nose.tools import raises
from pysegbase import graph

class ImageManipulationTest(unittest.TestCase):
    # @classmethod
    # def setUpClass(cls):
    #     if sys.version_info.major < 3:
    #         cls.assertCountEqual = cls.assertItemsEqual
    def test_resize_to_shape_no_new_unique_values(self):
        data = np.zeros([10, 15, 12])
        value1 = 1
        value2 = 2
        data[:5, :7, :6] = value1
        data[-5:, :7, :6] = value2

        expected_shape = [15, 15, 15]
        resized = imma.resize_to_shape(data, expected_shape)
        unique = np.unique(resized)

        self.assertEqual(resized.shape[0], expected_shape[0])
        self.assertEqual(resized.shape[1], expected_shape[1])
        self.assertEqual(resized.shape[2], expected_shape[2])
        self.assertEqual(resized[1, 1, 1], value1)
        self.assertEqual(resized[-2, 1, 1], value2)
        self.assertEqual(len(unique), 3)
        self.assertEqual(unique[0], 0)
        self.assertEqual(unique[1], 1)
        self.assertEqual(unique[2], 2)

    def test_resize_to_shape_wiht_zoom_no_new_unique_values(self):

        data = np.zeros([10, 15, 12])
        value1 = 1
        value2 = 2
        data[:5, :7, :6] = value1
        data[-5:, :7, :6] = value2

        expected_shape = [15, 15, 15]
        zoom = data.shape / np.array(expected_shape, dtype=float)
        resized = imma.resize_to_shape_with_zoom(data, expected_shape, zoom=zoom)
        unique = np.unique(resized)

        self.assertEqual(resized.shape[0], expected_shape[0])
        self.assertEqual(resized.shape[1], expected_shape[1])
        self.assertEqual(resized.shape[2], expected_shape[2])
        self.assertEqual(resized[1, 1, 1], value1)
        self.assertEqual(resized[-2, 1, 1], value2)
        self.assertEqual(len(unique), 3)
        self.assertEqual(unique[0], 0)
        self.assertEqual(unique[1], 1)
        self.assertEqual(unique[2], 2)

    def test_get_priority_objects(self):
        shape = [10, 15, 12]
        data = np.zeros(shape)
        value1 = 1
        value2 = 2
        data[:5, :7, :6] = value1
        data[-5:, :7, :6] = value2

        seeds = np.zeros(shape)
        seeds[9,3:6, 3] = 1

        selected = imma.select_objects_by_seeds(data, seeds)
        unique = np.unique(selected)
        #
        self.assertEqual(selected.shape[0], shape[0])
        self.assertEqual(selected.shape[1], shape[1])
        self.assertEqual(selected.shape[2], shape[2])
        self.assertEqual(selected[1, 1, 1], 1)
        self.assertEqual(selected[-2, 1, 1], 0)
        self.assertEqual(len(unique), 2)
        self.assertGreater(np.count_nonzero(data), np.count_nonzero(selected))

    def test_crop_and_uncrop(self):
        shape = [10, 10, 5]
        img_in = np.random.random(shape)

        crinfo = [[2, 8], [3, 9], [2, 5]]

        img_cropped = imma.crop(img_in, crinfo)

        img_uncropped = imma.uncrop(img_cropped, crinfo, shape)

        self.assertTrue(img_uncropped[4, 4, 3] == img_in[4, 4, 3])

    def test_multiple_crop_and_uncrop(self):
        """
        test combination of multiple crop
        """

        shape = [10, 10, 5]
        img_in = np.random.random(shape)

        crinfo1 = [[2, 8], [3, 9], [2, 5]]
        crinfo2 = [[2, 5], [1, 4], [1, 2]]

        img_cropped = imma.crop(img_in, crinfo1)
        img_cropped = imma.crop(img_cropped, crinfo2)

        crinfo_combined = imma.combinecrinfo(crinfo1, crinfo2)

        img_uncropped = imma.uncrop(img_cropped, crinfo_combined, shape)

        self.assertTrue(img_uncropped[4, 4, 3] == img_in[4, 4, 3])
        self.assertEquals(img_in.shape, img_uncropped.shape)
