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
from imcut import pycut


def fv_function(data, voxelsize, seeds=None, cls=None):
    """
    Creates feature vector for only data or for data from classes
    """

    fv1 = data.reshape(-1, 1)

    data2 = scipy.ndimage.filters.gaussian_filter(data, sigma=0.1)
    fv2 = data2.reshape(-1, 1)

    fv = np.hstack([fv1, fv2])

    if seeds is not None:
        logger.debug("seeds " + str(seeds))
        # print("seeds ", seeds)
        sd = seeds.reshape(-1, 1)
        selection = np.in1d(sd, cls)
        fv = fv[selection]
        sd = sd[selection]
        # sd = sd[]
        return fv, sd
    return fv


def box_data(noise_sigma=3):
    # data
    img3d = np.random.rand(32, 64, 64) * noise_sigma
    img3d[4:24, 12:32, 5:25] = img3d[4:24, 12:32, 5:25] + 30

    # seeds
    seeds = np.zeros([32, 64, 64], np.int8)
    seeds[9:12, 13:29, 18:25] = 1
    seeds[9:12, 4:9, 3:32] = 2
    # [mm]  10 x 10 x 10        # voxelsize_mm = [1, 4, 3]
    voxelsize_mm = [5, 5, 5]
    metadata = {"voxelsize_mm": voxelsize_mm}
    return img3d, seeds, voxelsize_mm


class PycutTest(unittest.TestCase):
    def setUp(self):
        self.segparams_ssgc = {
            # 'method':'graphcut',
            "method": "graphcut",
            "use_boundary_penalties": False,
            "boundary_dilatation_distance": 2,
            "boundary_penalties_weight": 1,
            "modelparams": {
                "type": "gmmsame",
                "params": {"n_components": 2},
                # "return_only_objects_with_seeds": True,
                # 'fv_type': "fv_extern",
                # 'fv_extern': fv_function,
                # 'adaptation': 'original_data',
            },
        }
        self.segparams_ssgc = {
            # 'method':'graphcut',
            "method": "graphcut",
            "use_boundary_penalties": False,
            "boundary_dilatation_distance": 2,
            "boundary_penalties_weight": 1,
            "modelparams": {
                "type": "gmmsame",
                "params": {"n_components": 2},
                # "return_only_objects_with_seeds": True,
                # 'fv_type': "fv_extern",
                # 'fv_extern': fv_function,
                # 'adaptation': 'original_data',
            },
        }
        self.segparams_lo2hi = {
            # 'method':'graphcut',
            "method": "graphcut",
            "use_boundary_penalties": False,
            "boundary_dilatation_distance": 2,
            "boundary_penalties_weight": 1,
            "modelparams": {
                "type": "gmmsame",
                "params": {"n_components": 2}
                # 'fv_type': "fv_extern",
                # 'fv_extern': fv_function,
                # 'adaptation': 'original_data',
            },
        }
        self.segparams_hi2lo = {
            # 'method':'graphcut',
            "method": "graphcut_hi2lo",
            "use_boundary_penalties": False,
            "boundary_dilatation_distance": 2,
            "boundary_penalties_weight": 1,
            "modelparams": {
                "type": "gmmsame",
                "params": {"n_components": 2}
                # 'fv_type': "fv_extern",
                # 'fv_extern': fv_function,
                # 'adaptation': 'original_data',
            },
        }

    @classmethod
    def setUpClass(cls):
        if sys.version_info.major < 3:
            cls.assertCountEqual = cls.assertItemsEqual

    @unittest.skipIf(
        os.environ.get("TRAVIS", True),
        "I dont know why this test fail on travis. Probably the RNG is allways the same",
    )
    def test_simple_graph_cut(self):
        img, seg, seeds = self.make_data(64, 20)
        gc = pycut.ImageGraphCut(img, segparams=self.segparams_ssgc)
        gc.set_seeds(seeds)

        gc.run()
        # import sed3
        # ed = sed3.sed3((gc.segmentation==0).astype(np.double), contour=seg)
        # ed.show()

        err = np.sum(
            np.abs((gc.segmentation == 0).astype(np.int8) - seg.astype(np.int8))
        )
        self.assertLess(err, 600)

    @attr("interactive")
    def test_simple_graph_cut_interactive(self):
        img, seg, seeds = self.make_data(64, 20)
        gc = pycut.ImageGraphCut(img, segparams=self.segparams_ssgc)
        gc.set_seeds(seeds)

        # gc.run()
        gc.interactivity()
        err = np.sum(
            np.abs((gc.segmentation == 0).astype(np.int8) - seg.astype(np.int8))
        )
        self.assertLess(err, 600)

    def test_simple_graph_cut_show_similarity(self):
        img, seg, seeds = self.make_data(64, 20)
        gc = pycut.ImageGraphCut(img, segparams=self.segparams_ssgc)
        gc.set_seeds(seeds)

        gc.run()
        gc.debug_show_reconstructed_similarity(show=False)
        gc.debug_show_model(start=-500, stop=500, show=False)
        # import sed3
        # ed = sed3.sed3((gc.segmentation==0).astype(np.double), contour=seg)
        # ed.show()

        err = np.sum(
            np.abs((gc.segmentation == 0).astype(np.int8) - seg.astype(np.int8))
        )
        self.assertLess(err, 600)

    def test_simple_graph_cut_feature_vector_fv001_intensity_and_blur(self):
        # test feature vector with
        segparams = {
            # 'method':'graphcut',
            "method": "graphcut",
            "use_boundary_penalties": False,
            "boundary_dilatation_distance": 2,
            "boundary_penalties_weight": 1,
            "modelparams": {
                "type": "gmmsame",
                "params": {"n_components": 2},
                "fv_type": "intensity_and_blur"
                # 'fv_type': "fv_extern",
                # 'fv_extern': fv_function,
                # 'adaptation': 'original_data',
            },
        }
        img, seg, seeds = self.make_data(64, 20)
        gc = pycut.ImageGraphCut(img, segparams=segparams)
        gc.set_seeds(seeds)

        gc.run()
        # outputs looks strange
        # gc.show_similarity(show=True)
        # gc.show_model(start=-500, stop=500, show=True)
        # import sed3
        # ed = sed3.sed3(img, contour=gc.segmentation)
        # ed.show()
        # ed = sed3.sed3((gc.segmentation==0).astype(np.double), contour=seg)
        # ed.show()

        err = np.sum(
            np.abs((gc.segmentation == 0).astype(np.int8) - seg.astype(np.int8))
        )
        # TODO check multidimensional feature vector. It looks to be not sufficient.
        # self.assertLess(err, 2000)

    def test_gc_box_overfiting(self):
        data3d, seeds, voxelsize = box_data(noise_sigma=0.5)
        segparams = {
            # 'method':'graphcut',
            "method": "graphcut",
            "use_boundary_penalties": False,
            "boundary_dilatation_distance": 2,
            "boundary_penalties_weight": 1,
            "modelparams": {
                "type": "gmmsame",
                "params": {"n_components": 3},
                # 'fv_type': "fv_extern",
                # 'fv_extern': fv_function,
                # 'adaptation': 'original_data',
            },
        }
        gc = pycut.ImageGraphCut(data3d, segparams=segparams, debug_images=False)
        gc.set_seeds(seeds)
        gc.run()
        # gc.interactivity()

        # import sed3
        # ed = sed3.sed3(data3d, contour=(gc.segmentation==0).astype(np.double) * 3)
        # ed.show()

    def test_simple_graph_cut_overfit_with_low_noise(self):
        img, seg, seeds = self.make_data(64, 20, sigma=20)
        segparams = {
            # 'method':'graphcut',
            "method": "graphcut",
            "use_boundary_penalties": False,
            "boundary_dilatation_distance": 2,
            "boundary_penalties_weight": 1,
            "modelparams": {
                "type": "gmmsame",
                "params": {"n_components": 3},
                # 'fv_type': "fv_extern",
                # 'fv_extern': fv_function,
                # 'adaptation': 'original_data',
            },
        }
        gc = pycut.ImageGraphCut(img, segparams=segparams, debug_images=False)
        gc.set_seeds(seeds)

        gc.run()
        # import sed3
        # ed = sed3.sed3(img, contour=(gc.segmentation==0).astype(np.double))
        # ed.show()

        err = np.sum(
            np.abs((gc.segmentation == 0).astype(np.int8) - seg.astype(np.int8))
        )
        self.assertLess(err, 600)

    @attr("interactive")
    def test_show_editor(self):
        """
        just run editor to see what is new
        Returns:

        """
        import imcut.seed_editor_qt
        import numpy as np
        from PyQt4.QtGui import QApplication

        app = QApplication(sys.argv)
        data = (np.random.rand(30, 31, 32) * 100).astype(np.int)
        data[15:40, 13:20, 10:18] += 50
        se = imcut.seed_editor_qt.QTSeedEditor(data)
        se.exec_()
        # self.assertTrue(False)

    # @attr('interactive')
    def test_draw_seed_function(self):
        """
        just run editor to see what is new
        Returns:

        """
        import imcut.seed_editor_qt
        import numpy as np
        from PyQt4.QtGui import QApplication

        app = QApplication(sys.argv)
        data = (np.random.rand(30, 31, 32) * 100).astype(np.int)
        data[15:40, 13:20, 10:18] += 50
        se = imcut.seed_editor_qt.QTSeedEditor(data)
        se.slice_box.seed_mark = 1  # left mouse button
        se.slice_box.last_position = [1, 3]
        se.slice_box.drawSeeds([10, 5])
        se.slice_box.seed_mark = 2  # left mouse button
        se.slice_box.last_position = [8, 1]
        se.slice_box.drawSeeds([7, 5])
        # se.exec_()

    # @TODO znovu zprovoznit test

    # @unittest.skip("Cekame, az to Tomas opravi")
    def make_data(self, sz=32, offset=0, sigma=80):
        seeds = np.zeros([sz, sz, sz], dtype=np.int8)
        seeds[offset + 12, offset + 9 : offset + 14, offset + 10] = 1
        seeds[offset + 20, offset + 18 : offset + 21, offset + 12] = 1
        img = np.ones([sz, sz, sz])
        img = img - seeds

        seeds[
            offset + 3 : offset + 15, offset + 2 : offset + 6, offset + 27 : offset + 29
        ] = 2
        img = scipy.ndimage.morphology.distance_transform_edt(img)
        segm = img < 7
        img = (100 * segm + sigma * np.random.random(img.shape)).astype(np.uint8)
        return img, segm, seeds

    def test_remove_repetitive(self):
        import time

        start = time.time()
        nlinks_not_unique = np.random.randint(0, 5, [100000, 3])
        nlinks = pycut.ms_remove_repetitive_link(nlinks_not_unique)
        elapsed = time.time() - start
        # print "elapsed ", elapsed

    def test_external_fv(self):
        """
        test external feature vector
        """

        img, seg, seeds = self.make_data(64, 20)
        segparams = {
            # 'method':'graphcut',
            "method": "graphcut",
            "use_boundary_penalties": False,
            "boundary_dilatation_distance": 2,
            "boundary_penalties_weight": 1,
            "modelparams": {"fv_type": "fv_extern", "fv_extern": fv_function},
        }
        gc = pycut.ImageGraphCut(img, segparams=segparams)
        gc.set_seeds(seeds)
        gc.run()
        # import sed3
        # ed = sed3.sed3(gc.segmentation==0, contour=seg)
        # ed.show()

        self.assertLess(
            np.sum(
                np.abs((gc.segmentation == 0).astype(np.int8) - seg.astype(np.int8))
            ),
            600,
        )

    def test_test_fv_function(self):
        img, seg, seeds = self.make_data(64, 20)
        vs = [1, 1, 1]
        out = fv_function(img, voxelsize=vs)
        print(out.shape)
        out1, out2 = fv_function(img, vs, seeds, [1, 2])
        print(np.min(out1), np.max(out1), out1.shape)
        print(np.min(out2), np.max(out2), out2.shape)
        self.assertEqual(out.shape[0], np.prod(img.shape))
        self.assertEqual(out1.shape[0], out2.shape[0])
        self.assertEqual(np.min(out2), 1)
        self.assertEqual(np.max(out2), 2)

    def test_external_fv_with_save(self):
        """
        test external feature vector with save model in the middle of processing
        """

        img, seg, seeds = self.make_data(64, 20)
        segparams = {
            # 'method':'graphcut',
            "method": "graphcut",
            "use_boundary_penalties": False,
            "boundary_dilatation_distance": 2,
            "boundary_penalties_weight": 1,
            "modelparams": {
                "type": "gmmsame",
                "fv_type": "fv_extern",
                "fv_extern": fv_function,
                "adaptation": "original_data",
            },
        }
        gc = pycut.ImageGraphCut(img, segparams=segparams)
        gc.set_seeds(seeds)

        gc.run()
        # import sed3
        # ed = sed3.sed3((gc.segmentation==0).astype(np.double), contour=seg)
        # ed.show()

        err = np.sum(
            np.abs((gc.segmentation == 0).astype(np.int8) - seg.astype(np.int8))
        )
        self.assertLess(err, 600)

        mdl_stored_file = "test_model.p"
        gc.save(mdl_stored_file)

        # forget
        gc = None

        img, seg, seeds = self.make_data(56, 18)
        # there is only one change in mdl params
        segparams["modelparams"] = {
            "mdl_stored_file": mdl_stored_file,
            "fv_extern": fv_function,
        }
        gc = pycut.ImageGraphCut(img, segparams=segparams)
        gc.set_seeds(seeds)
        gc.run()

        err = np.sum(
            np.abs((gc.segmentation == 0).astype(np.int8) - seg.astype(np.int8))
        )
        self.assertLess(err, 600)
        # import sed3
        # sed3.show_slices(img, contour=gc.segmentation==0, slice_step=6)

        # if we change the data there should be more error (assertMore)
        img = (img * 0.2).astype(np.uint8)
        # segparams['modelparams']['adaptation'] = 'original_data'
        # print(np.max(img))
        # print(np.min(img))
        gc = pycut.ImageGraphCut(img, segparams=segparams)
        gc.set_seeds(seeds)
        gc.run()

        m0 = gc.mdl.mdl[1]
        m1 = gc.mdl.mdl[2]
        logger.debug("model parameters")

        # import sed3
        # ed = sed3.sed3((gc.segmentation==0).astype(np.double), contour=seg)
        # ed.show()

        err = np.sum(
            np.abs((gc.segmentation == 0).astype(np.int8) - seg.astype(np.int8))
        )
        self.assertGreater(err, 600)
        # self.assertGreater(
        #     np.sum(
        #         np.abs(
        #             (gc.segmentation == 0).astype(np.int8) - seg.astype(np.int8))
        #     ),
        #     600)

        os.remove(mdl_stored_file)

    def test_apriori(self):
        """
        Test apriori segmentation. Make segmentation twice. First is used with gamma=0.2,
        second is used with gamma=0.9
        """

        img, seg, seeds = self.make_data(64, 20)
        apriori = np.zeros([64, 64, 64])

        apriori[:20, :20, :20] = 1
        segparams1 = {"apriori_gamma": 0.1}
        gc = pycut.ImageGraphCut(img, segparams=segparams1)
        gc.set_seeds(seeds)
        gc.apriori = apriori
        gc.run()
        # import sed3
        # ed = sed3.sed3(img, contour=(gc.segmentation==0))
        # ed.show()

        self.assertLess(
            np.sum(
                np.abs((gc.segmentation == 0).astype(np.int8) - seg.astype(np.int8))
            ),
            600,
        )

        segparams2 = {"apriori_gamma": 0.9}
        gc = pycut.ImageGraphCut(img, segparams=segparams2)
        gc.set_seeds(seeds)
        gc.apriori = apriori
        gc.run()

        self.assertLess(
            np.sum(
                np.abs((gc.segmentation == 0).astype(np.int8) - apriori.astype(np.int8))
            ),
            600,
        )

    def test_multiscale_gc_seg(self):
        """
        Test multiscale segmentation
        """

        img, seg, seeds = self.make_data(64, 20)
        segparams = {
            # 'method':'graphcut',
            "method": "multiscale_graphcut",
            "use_boundary_penalties": False,
            "boundary_dilatation_distance": 2,
            "boundary_penalties_weight": 1,
            "block_size": 8,
            "tile_zoom_constant": 1,
        }
        gc = pycut.ImageGraphCut(img, segparams=segparams)
        gc.set_seeds(seeds)
        gc.run()
        # import sed3
        # ed = sed3.sed3(gc.segmentation==0, contour=seg)
        # ed.show()

        self.assertLess(
            np.sum(
                np.abs((gc.segmentation == 0).astype(np.int8) - seg.astype(np.int8))
            ),
            600,
        )

    def test_fast_multiscale_gc_seg(self):
        """
        Test fast multiscale segmentation
        """

        img, seg, seeds = self.make_data(64, 20)
        segparams = {
            # 'method':'graphcut',
            "method": "multiscale_graphcut",
            "use_boundary_penalties": False,
            "boundary_dilatation_distance": 2,
            "boundary_penalties_weight": 1,
            "block_size": 8,
            "tile_zoom_constant": 1,
        }
        gc = pycut.ImageGraphCut(img, segparams=segparams)
        gc.set_seeds(seeds)
        gc.run()
        # import sed3
        # ed = sed3.sed3(gc.segmentation==0, contour=seg)
        # ed.show()

        self.assertLess(
            np.sum(
                np.abs((gc.segmentation == 0).astype(np.int8) - seg.astype(np.int8))
            ),
            600,
        )

    # @unittest.skip("Cekame, az to Mire opravi")
    def test_ms_seg_compared_with_different_resolution(self):
        """
        Test multiscale segmentation compared with different resolution
        """

        img, seg, seeds = self.make_data(64, 20)
        segparams = {
            # 'method':'graphcut',
            "method": "multiscale_graphcut_hi2lo",
            "use_boundary_penalties": False,
            "boundary_dilatation_distance": 2,
            "boundary_penalties_weight": 1,
            "block_size": 8,
            "tile_zoom_constant": 1,
        }
        gc = pycut.ImageGraphCut(img, segparams=segparams)
        gc.set_seeds(seeds)
        gc.run()
        # import sed3
        # ed = sed3.sed3(gc.segmentation==0, contour=seg)
        # ed.show()

        self.assertLess(
            np.sum(
                np.abs((gc.segmentation == 0).astype(np.int8) - seg.astype(np.int8))
            ),
            600,
        )

        # different resolution
        # sz = [128,128,128]
        sz = [70, 70, 70]
        sz = [90, 90, 90]
        sz = [100, 100, 100]
        sz = [200, 200, 200]
        sz1 = 70
        sz = [sz1, sz1, sz1]
        img2 = pycut.zoom_to_shape(img, sz, np.uint8)
        seg2 = pycut.zoom_to_shape(seg, sz, np.uint8)
        seeds2 = pycut.zoom_to_shape(seeds, sz, np.int8)

        segparams["tile_zoom_constant"] = 0.8
        gc = pycut.ImageGraphCut(img2, segparams=segparams)
        gc.set_seeds(seeds2)
        gc.run()
        # import sed3
        # ed = sed3.sed3(gc.segmentation==0, contour=seg2)
        # ed.show()

    def test_segmentation(self):
        img, seg, seeds = self.make_data()
        gc = pycut.ImageGraphCut(img)
        gc.set_seeds(seeds)
        gc.run()
        # import sed3
        # ed = sed3.sed3(gc.segmentation==0, contour=seg)
        # ed.show()
        self.assertLess(
            np.sum(
                np.abs((gc.segmentation == 0).astype(np.int8) - seg.astype(np.int8))
            ),
            30,
        )

    def test_multiscale_indexes(self):
        # there must be some data
        img = np.zeros([32, 32, 32], dtype=np.int16)
        gc = pycut.ImageGraphCut(img)

        # mask = np.zeros([1, 4, 4], dtype=np.int16)
        # mask[0,1:3,2:] = 1
        mask = np.zeros([1, 3, 3], dtype=np.int16)
        mask[0, 1:, 1] = 1
        orig_shape = [2, 6, 6]
        # zoom = 2

        inds, mask_resized = gc._ImageGraphCut__hi2lo_multiscale_indexes(
            mask, orig_shape
        )

        expected_result = [
            [
                [0, 0, 1, 1, 2, 2],
                [0, 0, 1, 1, 2, 2],
                [3, 3, 7, 8, 4, 4],
                [3, 3, 9, 10, 4, 4],
                [5, 5, 11, 12, 6, 6],
                [5, 5, 13, 14, 6, 6],
            ],
            [
                [0, 0, 1, 1, 2, 2],
                [0, 0, 1, 1, 2, 2],
                [3, 3, 15, 16, 4, 4],
                [3, 3, 17, 18, 4, 4],
                [5, 5, 19, 20, 6, 6],
                [5, 5, 21, 22, 6, 6],
            ],
        ]

        self.assertCountEqual(inds.reshape(-1), np.array(expected_result).reshape(-1))

    def test_ordered_values_by_indexes(self):
        """
        test of pycut.__ordered_values_by_indexes
        """

        # there must be some data
        img = np.zeros([32, 32, 32], dtype=np.int16)
        data = np.array([[0, 1, 1], [0, 2, 2], [0, 2, 2]])

        inds = np.array([[0, 1, 2], [3, 4, 4], [5, 4, 4]])
        gc = pycut.ImageGraphCut(img)
        vals = gc._ImageGraphCut__ordered_values_by_indexes(data, inds)
        expected = np.array([0, 1, 1, 0, 2, 0])
        self.assertCountEqual(vals, expected)

    def test_ordered_values_by_indexes_with_different_values(self):
        """
        test of pycut.__ordered_values_by_indexes
        in input data are non-consistent data
        Function should take maximal value
        """

        # there must be some data
        img = np.zeros([32, 32, 32], dtype=np.int16)
        data = np.array([[0, 1, 1], [0, 2, 2], [0, 3, 2]])

        inds = np.array([[0, 1, 2], [3, 4, 4], [5, 4, 4]])
        gc = pycut.ImageGraphCut(img)
        vals = gc._ImageGraphCut__ordered_values_by_indexes(data, inds)
        expected = np.array([0, 1, 1, 0, 3, 0])
        self.assertCountEqual(vals, expected)

    # @unittest.skip("This test is almost done")
    def test_msgc_lo2hi(self):
        """
        Test multiscale segmentation lo2hi
        """

        img, seg, seeds = self.make_data(64, 20)
        segparams = {
            # 'method':'graphcut',
            "method": "multiscale_graphcut_lo2hi",
            # 'method': 'multiscale_graphcut_hi2lo',
            "use_boundary_penalties": False,
            "boundary_dilatation_distance": 2,
            "boundary_penalties_weight": 1,
            "block_size": 8,
            "tile_zoom_constant": 1,
        }
        gc = pycut.ImageGraphCut(img, segparams=segparams, keep_graph_properties=True)
        gc.set_seeds(seeds)
        gc.run()

        edseeds = seeds
        # gc.interactive_inspect_node()

        # node_unariesalt, node_neighboor_edges_and_weights, node_neighboor_seeds = gc.inspect_node(edseeds)

        self.assertLess(
            np.sum(
                np.abs((gc.segmentation == 0).astype(np.int8) - seg.astype(np.int8))
            ),
            100,
        )

    def test_node_inspection_on_msgc_lo2hi(self):
        """
        Test multiscale segmentation lo2hi with node inspection
        """

        img, seg, seeds = self.make_data(64, 20)
        segparams = {
            # 'method':'graphcut',
            "method": "multiscale_graphcut_lo2hi",
            # 'method': 'multiscale_graphcut_hi2lo',
            "use_boundary_penalties": False,
            "boundary_dilatation_distance": 2,
            "boundary_penalties_weight": 1,
            "block_size": 8,
            "tile_zoom_constant": 1,
        }
        gc = pycut.ImageGraphCut(img, segparams=segparams, keep_graph_properties=True)
        gc.set_seeds(seeds)
        gc.run()

        self.assertLess(
            np.sum(
                np.abs((gc.segmentation == 0).astype(np.int8) - seg.astype(np.int8))
            ),
            100,
        )
        # gc.interactive_inspect_node()
        node_msindex = gc.debug_get_node_msindex(seeds)

        node_unariesalt, node_neighboor_edges_and_weights, node_neighboor_seeds = gc.debug_inspect_node(
            node_msindex=node_msindex
        )
        ii, ij, ik = np.nonzero(node_neighboor_seeds)
        #
        self.assertLessEqual(
            np.max(ii) - np.min(ii),
            24,
            msg="Neighbor nodes position variance should be maximum 3 * block size ",
        )
        self.assertLessEqual(
            np.max(ij) - np.min(ij),
            24,
            msg="Neighbor nodes position variance should be maximum 3 * block size ",
        )
        self.assertLessEqual(
            np.max(ik) - np.min(ik),
            24,
            msg="Neighbor nodes position variance should be maximum 3 * block size ",
        )

    def test_msgc_lo2hi_crazy_non_block_sized_images_65(self):
        """
        Test multiscale segmentation lo2hi crazy non block sized images 65
        """

        img, seg, seeds = self.make_data(65, 20)
        segparams = {
            # 'method':'graphcut',
            "method": "multiscale_graphcut_lo2hi",
            # 'method': 'multiscale_graphcut_hi2lo',
            "use_boundary_penalties": False,
            "boundary_dilatation_distance": 2,
            "boundary_penalties_weight": 1,
            "block_size": 8,
            "tile_zoom_constant": 1,
        }
        gc = pycut.ImageGraphCut(img, segparams=segparams, keep_graph_properties=True)
        gc.set_seeds(seeds)
        gc.run()

        edseeds = seeds
        # gc.interactive_inspect_node()

        # node_unariesalt, node_neighboor_edges_and_weights, node_neighboor_seeds = gc.inspect_node(edseeds)

        self.assertLess(
            np.sum(
                np.abs((gc.segmentation == 0).astype(np.int8) - seg.astype(np.int8))
            ),
            100,
        )

    def test_msgc_lo2hi_crazy_non_block_sized_images_68(self):
        """
        Test multiscale segmentation lo2hi crazy block 68
        """

        img, seg, seeds = self.make_data(68, 20)
        segparams = {
            # 'method':'graphcut',
            "method": "multiscale_graphcut_lo2hi",
            # 'method': 'multiscale_graphcut_hi2lo',
            "use_boundary_penalties": False,
            "boundary_dilatation_distance": 2,
            "boundary_penalties_weight": 1,
            "block_size": 8,
            "tile_zoom_constant": 1,
        }
        gc = pycut.ImageGraphCut(img, segparams=segparams, keep_graph_properties=True)
        gc.set_seeds(seeds)
        gc.run()

        edseeds = seeds
        # gc.interactive_inspect_node()

        # node_unariesalt, node_neighboor_edges_and_weights, node_neighboor_seeds = gc.inspect_node(edseeds)

        self.assertLess(
            np.sum(
                np.abs((gc.segmentation == 0).astype(np.int8) - seg.astype(np.int8))
            ),
            100,
        )

    def test_msgc_lo2hi_round_data(self):
        """
        Test multiscale segmentation lo2hi round data
        """
        np.random.seed(3)
        img, seg, seeds = generate_round_data(45, 3, 10, 3)
        blocksize = 6
        segparams = {
            # 'method':'graphcut',
            "method": "multiscale_graphcut_lo2hi",
            # 'method': 'multiscale_graphcut_hi2lo',
            "use_boundary_penalties": False,
            "boundary_dilatation_distance": 1,
            "boundary_penalties_weight": 1,
            "block_size": blocksize,
            "tile_zoom_constant": 1,
        }
        gc = pycut.ImageGraphCut(
            img,
            segparams=segparams,
            keep_graph_properties=True,
            # debug_images=True,
            debug_images=False,
        )
        gc.set_seeds(seeds)
        gc.run()
        # import sed3
        # ed = sed3.show_slices(img, gc.segmentation*2)

        # gc.debug_show_model()
        # gc.debug_show_reconstructed_similarity()
        tdata1, tdata2 = gc.debug_get_reconstructed_similarity()

        edseeds = seeds
        # node_unariesalt, node_neighboor_edges_and_weights, node_neighboor_seeds, node_msindex = gc.interactive_inspect_node()

        node_msindex = gc.debug_get_node_msindex(edseeds)
        node_unariesalt, node_neighboor_edges_and_weights, node_neighboor_seeds = gc.debug_inspect_node(
            node_msindex
        )

        nlinks_max = gc.debug_reconstruct_nlinks_max()

        from imcut import image_manipulation as imma

        seg_uncr = imma.uncrop(gc.segmentation, None, nlinks_max.shape)
        # import sed3
        # ed = sed3.sed3(nlinks_max, contour=seg_uncr*2)
        # ed.show()
        self.assertLess(
            np.sum(
                np.abs((gc.segmentation == 0).astype(np.int8) - seg.astype(np.int8))
            ),
            100,
        )
        self.assertEqual(
            np.max(nlinks_max),
            blocksize,
            "nlink maximal value in reconstructed nlink map. "
            "It should be size of block if there are at least some"
            " lowres blocks.",
        )
        self.assertEqual(
            np.min(nlinks_max), 1, "nlink minimal value in reconstructed nlink map"
        )
        self.assertGreaterEqual(
            np.min(tdata1), 0, "tlink minimal value in reconstructed tlink map"
        )
        self.assertGreaterEqual(
            np.min(tdata2), 0, "tlink minimal value in reconstructed tlink map"
        )
        self.assertGreaterEqual(
            node_neighboor_edges_and_weights.shape[0],
            3,
            "check number of nlink connections",
        )
        self.assertLessEqual(
            node_neighboor_edges_and_weights.shape[0],
            6,
            "check number of nlink connections",
        )
        self.assertGreaterEqual(
            np.min(node_unariesalt), 0, "selected node nlink minimum"
        )

    def test_ssgc_just_objects_with_seeds_round_data(self):
        """
        Test single scale graph cut just objects with seeds on round data
        """
        np.random.seed(3)
        img, seg, seeds = generate_round_data(
            45, 3, 10, 3, add_object_without_seeds=True
        )
        segparams = {
            "method": "graphcut",
            # 'method': 'multiscale_graphcut_hi2lo',
            "use_boundary_penalties": False,
            "boundary_dilatation_distance": 2,
            "boundary_penalties_weight": 1,
            "block_size": 8,
            "tile_zoom_constant": 1,
            "return_only_object_with_seeds": True,
        }
        gc = pycut.ImageGraphCut(img, segparams=segparams, keep_graph_properties=True)
        gc.set_seeds(seeds)
        gc.run()
        # import sed3
        # ed = sed3.show_slices(img, gc.segmentation*2, seeds=seeds)

        self.assertEqual((gc.segmentation == 0).astype(np.int8)[-3, -3, -3], 0)
        self.assertLess(
            np.sum(
                np.abs((gc.segmentation == 0).astype(np.int8) - seg.astype(np.int8))
            ),
            1500,
            msg="error is expected in the corner",
        )
        self.assertGreater(
            np.sum(
                np.abs((gc.segmentation == 0).astype(np.int8) - seg.astype(np.int8))
            ),
            1000,
            msg="error is expected in the corner",
        )

    def test_save_load(self):
        """
        test external feature vector with save model in the middle of processing
        """

        img, seg, seeds = self.make_data(64, 20)
        segparams = {
            # 'method':'graphcut',
            "method": "graphcut",
            "use_boundary_penalties": False,
            "boundary_dilatation_distance": 2,
            "boundary_penalties_weight": 1,
            "modelparams": {
                "type": "gmmsame",
                "fv_type": "intensity",
                # 'fv_extern': fv_function,
                "adaptation": "original_data",
            },
        }
        gc = pycut.ImageGraphCut(img, segparams=segparams)
        gc.set_seeds(seeds)

        gc.run()
        # import sed3
        # ed = sed3.sed3((gc.segmentation==0).astype(np.double), contour=seg)
        # ed.show()

        err = np.sum(
            np.abs((gc.segmentation == 0).astype(np.int8) - seg.astype(np.int8))
        )
        self.assertLess(err, 600)

        mdl_stored_file = "test_model.p"
        gc.save(mdl_stored_file)

        # forget
        gc = None

        img, seg, seeds = self.make_data(56, 18)
        # there is only one change in mdl params
        # segparams['modelparams'] = {
        #     'mdl_stored_file': mdl_stored_file,
        # }
        gc = pycut.ImageGraphCut(img)  # , segparams=segparams)
        gc.load(mdl_stored_file)
        gc.set_seeds(seeds)
        gc.run()

        err = np.sum(
            np.abs((gc.segmentation == 0).astype(np.int8) - seg.astype(np.int8))
        )
        self.assertLess(err, 600)
        # import sed3
        # sed3.show_slices(img, contour=gc.segmentation==0, slice_step=6)

        # if we change the data there should be more error (assertMore)
        img = (img * 0.2).astype(np.uint8)
        # segparams['modelparams']['adaptation'] = 'original_data'
        # print(np.max(img))
        # print(np.min(img))
        gc = pycut.ImageGraphCut(img)  # , segparams=segparams)
        gc.load(mdl_stored_file)
        gc.set_seeds(seeds)
        gc.run()

        m0 = gc.mdl.mdl[1]
        m1 = gc.mdl.mdl[2]
        logger.debug("model parameters")

        # import sed3
        # ed = sed3.sed3((gc.segmentation==0).astype(np.double), contour=seg)
        # ed.show()

        err = np.sum(
            np.abs((gc.segmentation == 0).astype(np.int8) - seg.astype(np.int8))
        )
        self.assertGreater(
            err, 600, "There should be greater error when we changed the data"
        )
        # self.assertGreater(
        #     np.sum(
        #         np.abs(
        #             (gc.segmentation == 0).astype(np.int8) - seg.astype(np.int8))
        #     ),
        #     600)

        os.remove(mdl_stored_file)


def generate_round_data(
    sz=32, offset=0, radius=7, seedsz=3, add_object_without_seeds=False
):
    # seedsz= int(sz/10)
    space = 2
    seeds = np.zeros([sz, sz + 1, sz + 2], dtype=np.int8)
    xmin = radius + seedsz + offset + 2
    ymin = radius + seedsz + offset + 6
    seeds[offset + 12, xmin + 3 : xmin + 7 + seedsz, ymin : ymin + 2] = 1
    seeds[offset + 20, xmin + 7 : xmin + 12 + seedsz, ymin + 5 : ymin + 7] = 1

    # add temp seed
    if add_object_without_seeds:
        seeds[-3, -3, -3] = 1
    img = np.ones([sz, sz + 1, sz + 2])
    img = img - seeds

    seeds[2 : 10 + seedsz, 2 : 9 + seedsz, 2 : 3 + seedsz] = 2

    # remove temo seed
    if add_object_without_seeds:
        seeds[-3, -3, -3] = 0

    img = scipy.ndimage.morphology.distance_transform_edt(img)
    segm = img < radius
    img = (100 * segm + 80 * np.random.random(img.shape)).astype(np.uint8)
    return img, segm, seeds


if __name__ == "__main__":
    unittest.main()
