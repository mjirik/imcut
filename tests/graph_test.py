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
    def test_graph_2d_implementation(self):

        data = np.array([[0,0,0,0,1,1,0],
                         [0,1,1,0,1,0,1],
                         [0,1,1,0,1,0,0],
                         [1,1,1,0,0,0,0],
                         [0,0,0,0,0,0,1]])
        g = graph.Graph(data, (0.1, 0.12), grid_function="2d", nsplit=3)
        # g = graph.Graph(data, (0.1, 0.12), grid_function="nd", nsplit=5)
        g.run(base_grid_vtk_fn="base_grid.vtk", final_grid_vtk_fn="final_grid.vtk")

    def test_graph_3d_implementation(self):

        data = np.array([[0,0,0,0,1,1,0],
                         [0,1,1,0,1,0,1],
                         [0,1,1,0,1,0,0],
                         [1,1,1,0,0,0,0],
                         [0,0,0,0,0,0,1]])
        # g = graph.Graph(data, (0.1, 0.12), grid_function="2d", nsplit=3)
        g = graph.Graph(data, (0.1, 0.12), grid_function="nd", nsplit=5)
        g.run(base_grid_vtk_fn="base_grid.vtk", final_grid_vtk_fn="final_grid.vtk")

    @unittest.skip("waiting for fix")
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
        g.run()

    # @unittest.skip("waiting for fix")
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
        g = graph.Graph(data, (0.1, 0.12, 0.05), grid_function="nd", nsplit=2)
        g.run(base_grid_vtk_fn="base_grid.vtk", final_grid_vtk_fn="final_grid.vtk")
        # g.run()

    def test_into_edges_reconnetcion_onsmall_graph_3d(self):

        data = np.array(
            [
                [[0,0,],
                 [0,0,]],
                [[0,0,],
                 [0,1,]],
            ]
        )
        g = graph.Graph(data, (0.1, 0.2, 0.05), grid_function="nd", nsplit=2)
        g.run(base_grid_vtk_fn="base_grid.vtk", final_grid_vtk_fn="final_grid.vtk")
        g.edges
        self.assertEqual(g.edges.shape[0], 33)
        self.assertEqual(g.edges.shape[1], 2)
        self.assertEqual(g.nodes.shape[0], 15)

    def test_from_edges_reconnetcion_onsmall_graph_3d(self):

        data = np.array(
            [
                [[1,0,],
                 [0,0,]],
                [[0,0,],
                 [0,0,]],
            ]
        )
        g = graph.Graph(data, (0.1, 0.2, 0.05), grid_function="nd", nsplit=2)
        g.run(base_grid_vtk_fn="base_grid.vtk", final_grid_vtk_fn="final_grid.vtk")
        self.assertEqual(g.edges.shape[0], 33)
        self.assertEqual(g.edges.shape[1], 2)
        self.assertEqual(g.nodes.shape[0], 15)

    def test_from_edges_reconnetcion_onsmall_graph_3d_higres_neighborhood(self):

        data = np.array(
            [
                [[1,1,],
                 [1,0,]],
                [[1,0,],
                 [0,0,]],
            ]
        )
        g = graph.Graph(data, (0.1, 0.2, 0.05), grid_function="nd", nsplit=2)
        g.run(base_grid_vtk_fn="base_grid.vtk", final_grid_vtk_fn="final_grid.vtk")
        # self.assertEqual(g.edges.shape[0], 33)
        # self.assertEqual(g.edges.shape[1], 2)
        # self.assertEqual(g.nodes.shape[0], 15)

    def test_into_edges_reconnetcion_onsmall_graph_3d_higres_neighborhood(self):

        data = np.array(
            [
                [[0,0,],
                 [0,1,]],
                [[0,1,],
                 [1,1,]],
            ]
        )
        g = graph.Graph(data, (0.1, 0.2, 0.05), grid_function="nd", nsplit=2)
        g.run(base_grid_vtk_fn="base_grid.vtk", final_grid_vtk_fn="final_grid.vtk")
        # self.assertEqual(g.edges.shape[0], 33)
        # self.assertEqual(g.edges.shape[1], 2)
        # self.assertEqual(g.nodes.shape[0], 15)

    def _test_automatic_ms_indexes_2d_same_as_orig(self, size):
        shape = [size, size]
        srt = graph.SRTab()
        subtab = srt.get_sr_subtab(shape)

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
        srt = graph.SRTab()
        subtab = srt.get_sr_subtab(shape)
        subtab

        # err = np.sum(np.abs(subtab - orig_sr_tab[size]))
        # self.assertEqual(err, 0)

    def test_gen_base_graph_2d(self):
        shape = [2, 3]
        voxelsize = [1., .6]
        # srt = graph.Graph(shape)

        nodes1, edges1, edg_dir1 = graph.gen_grid_2d(shape, voxelsize)
        nodes2, edges2, edg_dir2 = graph.gen_grid_nd(shape, voxelsize)

        graph.write_grid_to_vtk("grid1.vtk", nodes1, edges1)
        graph.write_grid_to_vtk("grid2.vtk", nodes2, edges2)
        nodes1


    def test_multiscale_index_set_lowres(self):
        shape = [2, 3]
        block_size = 2
        msi = graph.MultiscaleIndex(shape, block_size=block_size)
        # set first block
        msi.set_block_higres(0, 5)
        self.assertEqual(msi.msindex[0, 0], 5)
        # set last block
        msi.set_block_lowres(np.prod(shape) - 1, 3)
        self.assertEqual(msi.msindex[-1, -1], 3)


    def test_multiscale_index_set_higres(self):
        shape = [2, 3]
        block_size = 2
        msi = graph.MultiscaleIndex(shape, block_size=block_size)
        # set last block
        msi.set_block_higres(np.prod(shape) - 1, [11, 12, 13, 14])
        self.assertEqual(msi.msindex[-1, -1], 14)




if __name__ == "__main__":
    unittest.main()
