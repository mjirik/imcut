#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Organ segmentation

Example:

$ pycat -f head.mat -o brain.mat
"""

#import unittest
from optparse import OptionParser
import sys
import logging
logger = logging.getLogger(__name__)

from scipy.io import loadmat
import scipy.ndimage
import numpy as np

from pygco import cut_simple, cut_from_graph

import sklearn
import sklearn.mixture

# version comparison
from pkg_resources import parse_version

if parse_version(sklearn.__version__) > parse_version('0.10'):
    #new versions
    defaultmodelparams =  {'type':'gmmsame','params':{'covariance_type':'full'}}
else:
    defaultmodelparams =  {'type':'gmmsame','params':{'cvtype':'full'}}


class Model:
    """ Model for image intensity. Last dimension represent feature vector. 
    m = Model()
    m.train(cla, clb)
    X = numpy.random.random([2,3,4])
    # we have data 2x3 with fature vector with 4 fatures
    m.likelihood(X,0)
    """
    def __init__ (self, nObjects=2, modelparams=defaultmodelparams):

        self.mdl = {}
        self.modelparams = modelparams
        
    def train(self, clx, cl):
        """ Train clas number cl with data clx """

        if self.modelparams['type'] == 'gmmsame':
            gmmparams = self.modelparams['params']
            self.mdl[cl] = sklearn.mixture.GMM(**gmmparams)
            if len(clx.shape) == 1:
                # je to jen jednorozměrný vektor, tak je potřeba to převést na 2d matici
                clx = clx.reshape(-1,1)
            self.mdl[cl].fit(clx)

        else:
            raise NameError("Unknown model type")

        #pdb.set_trace();

    def likelihood(self, x, cl, onedimfv = True):
        """
        X = numpy.random.random([2,3,4])
        # we have data 2x3 with fature vector with 4 fatures
        m.likelihood(X,0)
        """

        sha = x.shape
        if onedimfv:
            xr = x.reshape(-1, 1)
        else:
            xr = x.reshape(-1, sha[-1])

        px = self.mdl[cl].score(xr)

#todo ošetřit více dimenzionální fv
        px = px.reshape(sha)
        return px

class ImageGraphCut:
    """
    Interactive Graph Cut

    ImageGraphCut(data, zoom, modelparams)
    scale

    Example:

    igc = ImageGraphCut(data)
    igc.interactivity()
    igc.make_gc()
    igc.show_segmentation()
    """
    def __init__(self, img, modelparams = defaultmodelparams,
                 gcparams = {'pairwiseAlpha':10}, voxelsize=None):

        self.img = img
        self.tdata = {}
        self.segmentation = []
        self.imgshape = img.shape
        self.modelparams = modelparams
        self.gcparams = gcparams
        self.seeds = np.zeros(self.img.shape, dtype=np.int8)

        if voxelsize is not None:
            self.voxel_volume = np.prod(voxelsize)

        else:
            self.voxel_volume = None

    def interactivity_loop(self, pyed):
        self.seeds = pyed.getSeeds()
        self.voxels1 = pyed.getSeedsVal(1)
        self.voxels2 = pyed.getSeedsVal(2)
        self.make_gc()
        pyed.setContours(1 - self.segmentation.astype(np.int8))

    def interactivity(self):
        """
        Interactive seed setting with 3d seed editor
        """
        from seed_editor_qt import QTSeedEditor
        from PyQt4.QtGui import QApplication

        app = QApplication(sys.argv)
        pyed = QTSeedEditor(self.img,
                            modeFun=self.interactivity_loop,
                            voxelVolume=self.voxel_volume)
        app.exec_()

    def make_gc(self):
        res_segm = self.set_data(self.img, self.voxels1, self.voxels2, seeds=self.seeds)

        self.segmentation = res_segm

    def set_hard_hard_constraints(self, tdata1, tdata2, seeds):
        tdata1[seeds==2] = np.max(tdata1) + 1
        tdata2[seeds==1] = np.max(tdata2) + 1
        tdata1[seeds==1] = 0
        tdata2[seeds==2] = 0

        return tdata1, tdata2

    def set_data(self, data, voxels1, voxels2, seeds = False, hard_constraints = True):
        """
        Setting of data.
        You need set seeds if you want use hard_constraints.
        
        """
        mdl = Model ( modelparams = self.modelparams )
        mdl.train(voxels1, 1)
        mdl.train(voxels2, 2)
        #pdb.set_trace();
        #tdata = {}
# as we convert to int, we need to multipy to get sensible values

# There is a need to have small vaues for good fit
# R(obj) = -ln( Pr (Ip | O) )
# R(bck) = -ln( Pr (Ip | B) )
# Boykov2001a 
# ln is computed in likelihood 
        tdata1 = (-(mdl.likelihood(data, 1))) * 10
        tdata2 = (-(mdl.likelihood(data, 2))) * 10

        if hard_constraints: 
            #pdb.set_trace();
            if (type(seeds)=='bool'):
                raise Exception ('Seeds variable  not set','There is need set seed if you want use hard constraints')
            tdata1, tdata2 = self.set_hard_hard_constraints(tdata1, tdata2, seeds)
            

        unariesalt = (1 * np.dstack([tdata1.reshape(-1,1), tdata2.reshape(-1,1)]).copy("C")).astype(np.int32)

# create potts pairwise
        #pairwiseAlpha = -10
        pairwise = -self.gcparams['pairwiseAlpha'] * np.eye(2, dtype=np.int32)
# use the gerneral graph algorithm
# first, we construct the grid graph
        inds = np.arange(data.size).reshape(data.shape)
        edgx = np.c_[inds[:, :, :-1].ravel(), inds[:, :, 1:].ravel()]
        edgy = np.c_[inds[:, :-1, :].ravel(), inds[:, 1:, :].ravel()]
        edgz = np.c_[inds[:-1, :, :].ravel(), inds[1:, :, :].ravel()]
        edges = np.vstack([edgx, edgy, edgz]).astype(np.int32)

# edges - seznam indexu hran, kteres spolu sousedi

# we flatten the unaries
        #result_graph = cut_from_graph(edges, unaries.reshape(-1, 2), pairwise)
        result_graph = cut_from_graph(edges, unariesalt.reshape(-1,2), pairwise)

        
        result_labeling = result_graph.reshape(data.shape)

        return result_labeling
                                                                                                        
# class Tests(unittest.TestCase):
#     def setUp(self):
#         pass

#     def test_segmentation(self):
#         data_shp = [16,16,16]
#         data = generate_data(data_shp)
#         seeds = np.zeros(data_shp)
# # setting background seeds
#         seeds[:,0,0] = 1
#         seeds[6,8:-5,2] = 2
#     #x[4:-4, 6:-2, 1:-6] = -1

#         igc = ImageGraphCut(data)
#         #igc.interactivity()
# # instead of interacitivity just set seeeds
#         igc.noninteractivity(seeds)

# # instead of showing just test results
#         #igc.show_segmentation()
#         segmentation = igc.segmentation
#         # Testin some pixels for result
#         self.assertTrue(segmentation[0, 0, -1] == 0)
#         self.assertTrue(segmentation[7, 9, 3] == 1)
#         self.assertTrue(np.sum(segmentation) > 10)
#         #pdb.set_trace()
#         #self.assertTrue(True)


#         #logger.debug(igc.segmentation.shape)

usage = '%prog [options]\n' + __doc__.rstrip()
help = {
    'in_file': 'input *.mat file with "data" field',
    'out_file': 'store the output matrix to the file',
    'debug': 'debug mode',
    'test': 'run unit test',
}

def main():
    logger = logging.getLogger()
    logger.setLevel(logging.WARNING)
    logging.basicConfig(format='%(message)s')
    ch = logging.StreamHandler()
    formatter = logging.Formatter("%(levelname)-5s [%(module)s:%(funcName)s:%(lineno)d] %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    parser = OptionParser(description='Organ segmentation')
    parser.add_option('-b','--debug', action='store_true',
                      dest='debug', help=help['debug'])
    parser.add_option('-f','--filename', action='store',
                      dest='in_filename', default=None,
                      help=help['in_file'])
    parser.add_option('-t', '--tests', action='store_true',
                      dest='unit_test', help=help['test'])
    parser.add_option('-o', '--outputfile', action='store',
                      dest='out_filename', default='output.mat',
                      help=help['out_file'])
    (options, args) = parser.parse_args()

    if options.debug:
        logger.setLevel(logging.DEBUG)

    # if options.tests:
    #     sys.argv[1:]=[]
    #     unittest.main()

    if options.in_filename is None:
        raise IOError('No input data!')

    else:
        dataraw = loadmat(options.in_filename,
                          variable_names=['data', 'voxelsizemm'])

    igc = ImageGraphCut(dataraw['data'], voxelsize=dataraw['voxelsizemm'])
    igc.interactivity()

    logger.debug(igc.segmentation.shape)

if __name__ == "__main__":
    main()
