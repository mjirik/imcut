#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
# import os.path as op

logger = logging.getLogger(__name__)
import numpy as nm
import numpy as np
import copy
from io import open

# TODO zpětná indexace původních pixelů (v add_nodes())
# TODO nastavení velikosti bloku (v sr_tab)
# TODO funguje to ve 3D?
# TODO možnost vypínání zápisu do VTK (mjirik)
# TODO možnost kontroly jmena souborů do VTK(mjirik)
#
# TODO Jeden nový uzel reprezentuje více voxelů v původním obrázku
# TODO indexy vytvářet ve split_voxel()?
# TODO co je igrp? split_voxel()
# TODO Uzly se jen přidávají, při odběru se nastaví flag na False? Dořešuje se to ve finish()
# TODO co je ndid ve split_voxel()
# TODO Přidat relabeling ve finish
# TODO zápis do VTK souboru? line 381, #  f = open(fname, 'w') # UnicodeDecodeError: 'ascii' codec can't decode byte 0xcd in position 0: ordinal not in range(128)
#
# data = nm.array([[0,0,0],
#                  [0,1,1],
#                  [0,1,1],
#                  [1,0,1]])

class Graph(object):

    # spliting reconnection table
    sr_tab = {
        2: nm.array([(0,2), (0,1), (1,3), (2,3)]),
        3: nm.array([(0,3,6), (0,1,2), (2,5,8), (6,7,8)]),
        4: nm.array([(0,4,8,12), (0,1,2,3), (3,7,11,15), (12,13,14,15)]),
    }

    def add_nodes(self, coors):
        """
        Add new nodes at the end of the list.
        """
        last = self.lastnode
        if type(coors) is nm.ndarray:
            if len(coors.shape) == 1:
                coors = coors.reshape((1,3))

            nadd = coors.shape[0]
            idx = slice(last, last + nadd)
        else:
            nadd = 1
            idx = self.lastnode

        self.nodes[idx,:] = coors
        self.node_flag[idx] = True
        self.lastnode += nadd
        self.nnodes += nadd

    def add_edges(self, conn, vert_flag, edge_group=None):
        """
        Add new edges at the end of the list.
        """
        last = self.lastedge
        if type(conn) is nm.ndarray:
            nadd = conn.shape[0]
            idx = slice(last, last + nadd)
            if edge_group is None:
                edge_group = nm.arange(nadd) + last
        else:
            nadd = 1
            idx = nm.array([last])
            conn = nm.array(conn).reshape((1,2))
            if edge_group is None:
                edge_group = idx

        self.edges[idx,:] = conn
        self.edge_flag[idx] = True
        self.edge_dir[idx] = vert_flag
        self.edge_group[idx] = edge_group
        self.lastedge += nadd
        self.nedges += nadd

    def finish(self):
        ndidxs = nm.where(self.node_flag)[0]
        aux = - nm.ones((self.nodes.shape[0],), dtype=nm.int16)
        aux[ndidxs] = nm.arange(ndidxs.shape[0])
        edges = aux[self.edges[self.edge_flag]]
        nodes = self.nodes[ndidxs]

        del self.nodes
        del self.node_flag
        del self.edges
        del self.edge_flag
        del self.edge_dir
        del self.edge_group

        self.nodes = nodes
        self.edges = edges
        self.node_flag = nm.ones((nodes.shape[0],), dtype=nm.bool)
        self.edge_flag = nm.ones((edges.shape[0],), dtype=nm.bool)

    def write_vtk(self, fname):
        write_grid_to_vtk(self.nodes, self.edges, self.node_flag, self.edge_flag)

    def edges_by_group(self, idxs):
        """

        :param idxs: množina hran
        :return:
        """
        ed = self.edge_group[idxs]
        ugrps = nm.unique(ed)
        out = []
        for igrp in ugrps:
            out.append(idxs[nm.where(ed == igrp)[0]])

        return out

    def split_voxel(self, ndid, nsplit):
        """

        :param ndid:
        :param tile_shape:
        :return:
        """
        # nsplit - was size of split square, tiles_shape = [nsplit, nsplit]
        # generate subgrid
        # tile_shape = tuple(tile_shape)
        # TODO remove
        # nsplit = tile_shape[0]
        tile_shape = (nsplit, nsplit)
        if tile_shape in self.cache:
            nd, ed, ed_dir = self.cache[tile_shape]
        else:
            nd, ed, ed_dir = gen_base_graph(tile_shape, self.voxelsize / nsplit)
            # nd, ed, ed_dir = gen_base_graph(tile_shape, self.voxelsize / tile_shape)
            self.cache[tile_shape] = nd, ed, ed_dir

        ndoffset = self.lastnode
        self.add_nodes(nd + self.nodes[ndid,:] - (self.voxelsize / 2))
        self.add_edges(ed + ndoffset, ed_dir)

        # connect subgrid
        ed_remove = []
        # sr_tab_old = self.sr_tab[nsplit]
        srt = SRTab(tile_shape)
        sr_tab = srt.get_sr_subtab()
        idxs = nm.where(self.edge_flag > 0)[0]

        # edges "into" node?
        eidxs = idxs[nm.where(self.edges[idxs,1] == ndid)[0]]
        for igrp in self.edges_by_group(eidxs):
            if igrp.shape[0] > 1:
                self.edges[igrp,1] = sr_tab[self.edge_dir[igrp[0]],:].T.flatten()\
                    + ndoffset
            else:
                ed_remove.append(igrp[0])
                muleidxs = nm.tile(igrp, nsplit)
                newed = self.edges[muleidxs,:]
                neweddir = self.edge_dir[muleidxs]
                newed[:,1] = sr_tab[self.edge_dir[igrp],:].T.flatten()\
                    + ndoffset
                self.add_edges(newed, neweddir, self.edge_group[igrp])

        # edges "from" node?
        eidxs = idxs[nm.where(self.edges[idxs,0] == ndid)[0]]
        for igrp in self.edges_by_group(eidxs):
            if igrp.shape[0] > 1:
                self.edges[igrp,1] = sr_tab[self.edge_dir[igrp[0]],:].T.flatten()\
                + ndoffset
            else:
                ed_remove.append(igrp[0])
                muleidxs = nm.tile(igrp, nsplit)
                newed = self.edges[muleidxs,:]
                neweddir = self.edge_dir[muleidxs]
                newed[:,0] = sr_tab[self.edge_dir[igrp] + 2,:].T.flatten()\
                    + ndoffset
                self.add_edges(newed, neweddir, self.edge_group[igrp])

        # remove node
        self.node_flag[ndid] = False
        # remove edges
        self.edge_flag[ed_remove] = False

    def generate_base_grid(self, vtk_filename=None):
        """
        Run first step of algorithm. Next step is split_voxels
        :param vtk_filename:
        :return:
        """
        nd, ed, ed_dir = gen_base_graph(self.data.shape, self.voxelsize)
        self.add_nodes(nd)
        self.add_edges(ed, ed_dir)

        if vtk_filename is not None:
            self.write_vtk(vtk_filename)

    def split_voxels(self, vtk_filename=None):
        """
        Second step of algorithm
        :return:
        """
        self.cache = {}
        idxs = nm.where(self.data)
        nr, nc = self.data.shape
        for k, (ir, ic) in enumerate(zip(*idxs)):
            ndid = ic + ir * nc
            self.split_voxel(ndid, 4)

        self.finish()
        if vtk_filename is not None:
            self.write_vtk(vtk_filename)

    def run(self):
        # cache dict.
        self.cache = {}

        # generate base grid
        # self.generate_base_grid("base_grid.vtk")
        self.generate_base_grid()
        # split voxels
        # self.split_voxels("final_grid.vtk")
        self.split_voxels()


    def __init__(self, data, voxelsize, dim=2, ndmax=400):
        self.dim = dim
        self.voxelsize = nm.asarray(voxelsize)

        # init nodes
        self.nnodes = 0
        self.lastnode = 0
        self.nodes = nm.zeros((ndmax, 3), dtype=nm.float32)
        # node_flag: if true, this node is used in final output
        self.node_flag = nm.zeros((ndmax,), dtype=nm.bool)

        # init edges
        edmax = 2 * ndmax
        self.nedges = 0
        self.lastedge = 0
        self.edges = - nm.ones((edmax, 2), dtype=nm.int16)
        # edge_flag: if true, this edge is used in final output
        self.edge_flag = nm.zeros((edmax,), dtype=nm.bool)
        self.edge_dir = nm.zeros((edmax,), dtype=nm.int8)
        self.edge_group = - nm.ones((edmax,), dtype=nm.int16)
        self.data = data



class SRTab(object):
    """
    Table connection on transition between low resolution and high resolution
    """
    def __init__(self, shape):
        self.shape = shape
        self.sr_tab = {}
        if len(shape) not in (2, 3):
            logger.error("2D or 3D shape expected")
        pass

    def get_sr_subtab(self):
        # direction_order = [0, 1, 2, 3, 4, 5, 6]
        inds = np.array(range(np.prod(self.shape)))
        reshaped = inds.reshape(self.shape)

        tab = []
        for direction in range(len(self.shape) - 1, -1, -1):
            # direction = direction_order[i]
            tab.append(reshaped.take(0, direction).flatten())
        for direction in range(len(self.shape) - 1, -1, -1):
            # direction = direction_order[i]
            tab.append(reshaped.take(-1, direction).flatten())
        return np.array(tab)

def grid_edges(shape, inds=None, return_directions=True):
    """
    Get list of grid edges
    :param shape:
    :param inds:
    :param return_directions:
    :return:
    """
    if inds is None:
        inds = np.arange(np.prod(shape)).reshape(shape)
    # if not self.segparams['use_boundary_penalties'] and \
    #         boundary_penalties_fcn is None :
    if len(shape) == 2:
        edgx = np.c_[inds[:, :-1].ravel(), inds[:, 1:].ravel()]
        edgy = np.c_[inds[:-1, :].ravel(), inds[1:, :].ravel()]

        edges = [ edgx, edgy]

        directions = [
            np.ones([edgx.shape[0]], dtype=np.int8) * 0,
            np.ones([edgy.shape[0]], dtype=np.int8) * 1,
            ]


    elif len(shape) == 3:
        # This is faster for some specific format
        edgx = np.c_[inds[:, :, :-1].ravel(), inds[:, :, 1:].ravel()]
        edgy = np.c_[inds[:, :-1, :].ravel(), inds[:, 1:, :].ravel()]
        edgz = np.c_[inds[:-1, :, :].ravel(), inds[1:, :, :].ravel()]
        edges = [ edgx, edgy, edgz]
    else:
        logger.error("Expected 2D or 3D data")

    # for all edges along first direction put 0, for second direction put 1, for third direction put 3
    if return_directions:
        directions = []
        for idirection in range(len(shape)):
            directions.append(
                np.ones([edges[idirection].shape[0]], dtype=np.int8) * idirection
            )
    edges = np.concatenate(edges)
    if return_directions:
        edge_dir = np.concatenate(directions)
        return edges, edge_dir
    else:
        return edges

def grid_nodes(shape, voxelsize=None):
    voxelsize = np.asarray(voxelsize)
    nodes = np.moveaxis(np.indices(shape), 0, -1).reshape(-1, len(shape))
    if voxelsize is not None:
        nodes = (nodes * voxelsize) + (0.5 * voxelsize)
    return nodes

def gen_base_graph_new(shape, voxelsize=None, inds=None):
    edges, edge_dir = grid_edges(shape, inds, return_directions=True)
    # nodes coordinates
    nodes = grid_nodes(shape, voxelsize)
    return nodes, edges, edge_dir

def gen_base_graph(shape, voxelsize):
    """
    Generate list of edges for a base grid.
    """
    nr, nc = shape
    nrm1, ncm1 = nr - 1, nc - 1
    # sh = nm.asarray(shape)
    # calculate number of edges, in 2D: (nrows * (ncols - 1)) + ((nrows - 1) * ncols)
    nedges = 0
    for direction in range(len(shape)):
        sh = copy.copy(list(shape))
        sh[direction] += -1
        nedges += nm.prod(sh)


    nedges_old = ncm1 * nr + nrm1 * nc
    edges = nm.zeros((nedges, 2), dtype=nm.int16)
    edge_dir = nm.zeros((ncm1 * nr + nrm1 * nc, ), dtype=nm.bool)
    nodes = nm.zeros((nm.prod(shape), 3), dtype=nm.float32)

    # edges
    idx = 0
    row = nm.zeros((ncm1, 2), dtype=nm.int16)
    row[:,0] = nm.arange(ncm1)
    row[:,1] = nm.arange(ncm1) + 1
    for ii in range(nr):
        edges[slice(idx, idx + ncm1),:] = row + nc * ii
        idx += ncm1

    edge_dir[slice(0, idx)] = 0 # horizontal dir

    idx0 = idx
    col = nm.zeros((nrm1, 2), dtype=nm.int16)
    col[:,0] = nm.arange(nrm1) * nc
    col[:,1] = nm.arange(nrm1) * nc + nc
    for ii in range(nc):
        edges[slice(idx, idx + nrm1),:] = col + ii
        idx += nrm1

    edge_dir[slice(idx0, idx)] = 1 # vertical dir

    # nodes
    idx = 0
    row = nm.zeros((nc, 3), dtype=nm.float32)
    row[:,0] = voxelsize[0] * (nm.arange(nc) + 0.5)
    row[:,1] = voxelsize[1] * 0.5
    for ii in range(nr):
        nodes[slice(idx, idx + nc),:] = row
        row[:,1] += voxelsize[1]
        idx += nc

    return nodes, edges, edge_dir

def write_grid_to_vtk(fname, nodes, edges, node_flag=None, edge_flag=None):
    """
    Write nodes and edges to VTK file
    :param fname: VTK filename
    :param nodes:
    :param edges:
    :param node_flag: set if this node is really used in output
    :param edge_flag: set if this flag is used in output
    :return:
    """

    if node_flag is None:
        node_flag = np.ones([nodes.shape[0]], dtype=np.bool)
    if edge_flag is None:
        edge_flag = np.ones([edges.shape[0]], dtype=np.bool)

    if nodes.shape[1] == 2:
        zeros = np.zeros([nodes.shape[0], 1], dtype=nodes.dtype)
        nodes = np.concatenate([nodes, zeros], axis=1)
    f = open(fname, 'w', encoding="utf-8")
    f.write('# vtk DataFile Version 2.6\n')
    f.write('output file\nASCII\nDATASET UNSTRUCTURED_GRID\n')

    idxs = nm.where(node_flag > 0)[0]
    nnd = len(idxs)
    aux = -nm.ones(node_flag.shape, dtype=nm.int32)
    aux[idxs] = nm.arange(nnd, dtype=nm.int32)
    f.write('\nPOINTS %d float\n' % nnd)
    for ndi in idxs:
        f.write('%.6f %.6f %.6f\n' % tuple(nodes[ndi,:]))

    idxs = nm.where(edge_flag > 0)[0]
    ned = len(idxs)
    f.write('\nCELLS %d %d\n' % (ned, ned * 3))
    for edi in idxs:
        f.write('2 %d %d\n' % tuple(aux[edges[edi,:]]))

    f.write('\nCELL_TYPES %d\n' % ned)
    for edi in idxs:
        f.write('3\n')

