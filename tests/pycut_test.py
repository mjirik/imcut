#! /usr/bin/python
# -*- coding: utf-8 -*-

# import funkcí z jiného adresáře
import sys
import os.path

path_to_script = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(path_to_script, "../src/"))
import unittest
import scipy

import numpy as np

from pysegbase import pycut


class PycutTest(unittest.TestCase):

    # @TODO znovu zprovoznit test

    # @unittest.skip("Cekame, az to Tomas opravi")
    def make_data(self, sz=32, offset=0):
        seeds = np.zeros([sz, sz, sz], dtype=np.int8)
        seeds[offset + 12, offset + 10:offset + 13, offset + 10] = 1
        seeds[offset + 20, offset + 18:offset + 21, offset + 12] = 1
        img = np.ones([sz, sz, sz])
        img = img - seeds

        seeds[
            offset + 3:offset + 15, 
            offset + 2:offset + 6, 
            offset + 27:offset + 29] = 2
        img = scipy.ndimage.morphology.distance_transform_edt(img)
        segm = img < 7
        img = (100 * segm + 80 * np.random.random(img.shape)).astype(np.uint8)
        return img, segm, seeds

    def test_remove_repetitive(self):
        import time
        start = time.time()
        nlinks_not_unique = np.random.randint(0, 5, [100000, 3])
        nlinks = pycut.ms_remove_repetitive_link(nlinks_not_unique)
        elapsed = (time.time() - start)
        print "elapsed ", elapsed
    # @unittest.skip("Cekame, az to Mire opravi")
    def test_ms_seg(self):
        """
        Test multiscale segmentation
        """

        img, seg, seeds = self.make_data(64, 20)
        segparams = {
                # 'method':'graphcut',
                'method':'multiscale_graphcut',
                'use_boundary_penalties': False,
                'boundary_dilatation_distance': 2,
                'boundary_penalties_weight': 1,
                'block_size': 8,
                'tile_zoom_constant': 1
                }
        gc = pycut.ImageGraphCut(img, segparams=segparams)
        gc.set_seeds(seeds)
        gc.run()
        # import sed3
        # ed = sed3.sed3(gc.segmentation==0, contour=seg)
        # ed.show()

        self.assertLess(
                np.sum(
                    np.abs(
                        (gc.segmentation == 0).astype(np.int8) - 
                        seg.astype(np.int8))
                    )
                , 600)


# different resolution
        # sz = [128,128,128]
        sz = [70,70,70]
        sz = [90,90,90]
        sz = [100,100,100]
        sz = [200,200,200]
        sz1 = 70
        sz = [sz1, sz1, sz1]
        img2 = pycut.zoom_to_shape(img, sz, np.uint8)
        seg2 = pycut.zoom_to_shape(seg, sz, np.uint8)
        seeds2 = pycut.zoom_to_shape(seeds, sz, np.int8)

        segparams['tile_zoom_constant'] = 0.8
        gc = pycut.ImageGraphCut(img2, segparams=segparams)
        gc.set_seeds(seeds2)
        gc.run()
        import sed3
        ed = sed3.sed3(gc.segmentation==0, contour=seg2)
        ed.show()
        

    def test_segmentation(self):
        img, seg, seeds = self.make_data()
        gc = pycut.ImageGraphCut(img)
        gc.set_seeds(seeds)
        gc.run()
        self.assertLess(
                np.sum(
                    np.abs(
                        (gc.segmentation == 0).astype(np.int8) - 
                        seg.astype(np.int8))
                    )
                , 30)
        

    def test_multiscale_indexes(self):
        # there must be some data
        img = np.zeros([32, 32, 32], dtype=np.int16)
        gc = pycut.ImageGraphCut(img)

        # mask = np.zeros([1, 4, 4], dtype=np.int16)
        # mask[0,1:3,2:] = 1
        mask = np.zeros([1, 3, 3], dtype=np.int16)
        mask[0, 1:, 1] = 1
        orig_shape = [2, 6, 6]
        zoom = 2

        inds = gc._ImageGraphCut__multiscale_indexes(mask, orig_shape, zoom)

        expected_result = [[[0, 0,  1,  1, 2, 2],
                            [0, 0,  1,  1, 2, 2],
                            [3, 3,  7,  8, 4, 4],
                            [3, 3,  9, 10, 4, 4],
                            [5, 5, 11, 12, 6, 6],
                            [5, 5, 13, 14, 6, 6]],
                           [[0, 0,  1,  1, 2, 2],
                            [0, 0,  1,  1, 2, 2],
                            [3, 3, 15, 16, 4, 4],
                            [3, 3, 17, 18, 4, 4],
                            [5, 5, 19, 20, 6, 6],
                            [5, 5, 21, 22, 6, 6]]]

        self.assertItemsEqual(inds.reshape(-1),
                              np.array(expected_result).reshape(-1))

    def test_ordered_values_by_indexes(self):
        """
        test of pycut.__ordered_values_by_indexes
        """

        # there must be some data
        img = np.zeros([32, 32, 32], dtype=np.int16)
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

    def test_ordered_values_by_indexes_with_different_values(self):
        """
        test of pycut.__ordered_values_by_indexes
        in input data are non-consistent data
        Function should take maximal value
        """

        # there must be some data
        img = np.zeros([32, 32, 32], dtype=np.int16)
        data = np.array([
            [0, 1, 1],
            [0, 2, 2],
            [0, 3, 2]
        ])

        inds = np.array([
            [0, 1, 2],
            [3, 4, 4],
            [5, 4, 4]
        ])
        gc = pycut.ImageGraphCut(img)
        vals = gc._ImageGraphCut__ordered_values_by_indexes(data, inds)
        expected = np.array([0, 1, 1, 0, 3, 0])
        self.assertItemsEqual(vals, expected)


if __name__ == "__main__":
    unittest.main()
