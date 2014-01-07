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
import numpy as np

from pygco import cut_from_graph

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
        """ Train clas number cl with data clx.

        clx: data, 2d matrix
        cl: label, integer
        """

        if len(clx.shape) == 1:
     #  je to jen jednorozměrný vektor, tak je potřeba to převést na 2d matici
            clx = clx.reshape(-1, 1)
        if self.modelparams['type'] == 'gmmsame':
            gmmparams = self.modelparams['params']
            self.mdl[cl] = sklearn.mixture.GMM(**gmmparams)
            self.mdl[cl].fit(clx)

        elif self.modelparams['type'] == 'kernel':
            kernelmodelparams = {'kernel': 'gaussian', 'bandwidth': 0.2}
            self.mdl[cl] = KernelDensity(**kernelmodelparams).fit(clx)
        else:
            raise NameError("Unknown model type")

        #pdb.set_trace();

    def likelihood(self, x, cl, onedimfv=True):
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

        if self.modelparams['type'] == 'gmmsame':
            px = self.mdl[cl].score(xr)

#todo ošetřit více dimenzionální fv
            px = px.reshape(sha)
        elif self.modelparams['type'] == 'kernel':
            px = self.mdl[cl].score_samples(xr)
        return px


class ImageGraphCut:

    """
    Interactive Graph Cut.

    ImageGraphCut(data, zoom, modelparams)
    scale

    Example:

    igc = ImageGraphCut(data)
    igc.interactivity()
    igc.make_gc()
    igc.show_segmentation()

    """

    def __init__(self, img,
                 modelparams=defaultmodelparams,
                 segparams={},
                 voxelsize=None):

# default values                              use_boundary_penalties
        #self.segparams = {'pairwiseAlpha':10, 'use_boundary_penalties':False}
        self.segparams = {'type':'graphcut',
                'pairwise_alpha':20,
                'use_boundary_penalties':False,
                'boundary_penalties_sigma':200,
                'boundary_penalties_weight':30}
        self.segparams.update(segparams)

        self.img = img
        self.tdata = {}
        self.segmentation = None
        self.imgshape = img.shape
        self.modelparams = modelparams
        #self.segparams = segparams
        self.seeds = np.zeros(self.img.shape, dtype=np.int8)

        self.voxelsize = voxelsize
        if voxelsize is not None:
            self.voxel_volume = np.prod(voxelsize)

        else:
            self.voxel_volume = None

    def interactivity_loop(self, pyed):
# @TODO stálo by za to, přehodit tlačítka na myši. Levé má teď jedničku,
# pravé dvojku. Pravým však zpravidla označujeme pozadí a tak nám vyjde
# popředí jako nula a pozadí jako jednička.
# Tím také dopadne jinak interaktivní a neinteraktivní varianta.
        self.seeds = pyed.getSeeds()
        self.voxels1 = pyed.getSeedsVal(1)
        self.voxels2 = pyed.getSeedsVal(2)
        self.make_gc()
        pyed.setContours(1 - self.segmentation.astype(np.int8))

        try:
            import audiosupport
            audiosupport.beep()
        except:
            print("cannot open audiosupport")

    def interactivity(self, min_val=None, max_val=None, qt_app=None):
        """
        Interactive seed setting with 3d seed editor
        """
        from seed_editor_qt import QTSeedEditor
        from PyQt4.QtGui import QApplication
        if min_val is None:
            min_val = np.min(self.img)

        if max_val is None:
            max_val = np.max(self.img)

        window_c = ((max_val + min_val)/2)#.astype(np.int16)
        window_w = (max_val - min_val)#.astype(np.int16)

        if qt_app is None:
            qt_app = QApplication(sys.argv)

        pyed = QTSeedEditor(self.img,
                            modeFun=self.interactivity_loop,
                            voxelSize=self.voxelsize,
                            seeds=self.seeds)

        pyed.changeC(window_c)
        pyed.changeW(window_w)

        qt_app.exec_()

    def set_seeds(self,seeds):
        """
        Function for manual seed setting. Sets variable seeds and prepares
        voxels for density model.
        """
        if self.img.shape != seeds.shape:
            raise Exception("Seeds must be same size as input image")

        self.seeds = seeds.astype('int8')
        self.voxels1 = self.img[self.seeds == 1]
        self.voxels2 = self.img[self.seeds == 2]

    def make_gc(self):
        res_segm = self.set_data(self.img,
                                 self.voxels1, self.voxels2,
                                 seeds=self.seeds)

        self.segmentation = res_segm

    def set_hard_hard_constraints(self, tdata1, tdata2, seeds):
        tdata1[seeds==2] = np.max(tdata1) + 1
        tdata2[seeds==1] = np.max(tdata2) + 1
        tdata1[seeds==1] = 0
        tdata2[seeds==2] = 0

        return tdata1, tdata2

    def boundary_penalties_array(self, axis, sigma = None):

        import scipy.ndimage.filters as scf

        #for axis in range(0,dim):
        filtered = scf.prewitt(self.img, axis=axis)
        if sigma is None:
            sigma2 = np.var(self.img)
        else:
            sigma2 = sigma**2


        filtered = np.exp(-np.power(filtered,2)/(256*sigma2))

        #srovnán hodnot tak, aby to vycházelo mezi 0 a 100
        #cc = 10
        #filtered = ((filtered - 1)*cc) + 10
        print 'ax %.1g max %.3g min %.3g  avg %.3g' % (
                axis,
                np.max(filtered), np.min(filtered), np.mean(filtered))
#
## @TODO Check why forumla with exp is not stable
## Oproti Boykov2001b tady nedělím dvojkou. Ta je tam jen proto,
## aby to slušně vycházelo, takže jsem si jí upravil
## Originální vzorec je
## Bpq = exp( - (Ip - Iq)^2 / (2 * \sigma^2) ) * 1 / dist(p,q)
#        filtered = (-np.power(filtered,2)/(16*sigma))
## Přičítám tu 256 což je empiricky zjištěná hodnota - aby to dobře vyšlo
## nedávám to do exponenciely, protože je to numericky nestabilní
#        filtered = filtered + 255 # - np.min(filtered2) + 1e-30
## Ještě by tady měl a následovat exponenciela, ale s ní je to numericky
## nestabilní. Netuším proč.
#        #if dim >= 1:
## odecitame od sebe tentyz obrazek
##            df0 = self.img[:-1,:] - self.img[]
##            diffs.insert(0,
        return filtered

    def set_data(self, data, voxels1, voxels2,
                 seeds=False,
                 hard_constraints=True,
                 area_weight=1):
        """
        Setting of data.
        You need set seeds if you want use hard_constraints.
        """

        # Dobře to fungovalo area_weight = 0.05 a cc = 6 a diference se
        # počítaly z :-1
        mdl = Model ( modelparams = self.modelparams )
        mdl.train(voxels1, 1)
        mdl.train(voxels2, 2)
        #pdb.set_trace();
        #tdata = {}
# as we convert to int, we need to multipy to get sensible values

# There is a need to have small vaues for good fit
# R(obj) = -ln( Pr (Ip | O) )
# R(bck) = -ln( Pr (Ip | B) )
# Boykov2001b
# ln is computed in likelihood
        tdata1 = (-(mdl.likelihood(data, 1))) * 10
        tdata2 = (-(mdl.likelihood(data, 2))) * 10

        if hard_constraints:
            #pdb.set_trace();
            if (type(seeds)=='bool'):
                raise Exception ('Seeds variable  not set','There is need set seed if you want use hard constraints')
            tdata1, tdata2 = self.set_hard_hard_constraints(tdata1,
                                                            tdata2,
                                                            seeds)


        unariesalt = (0+(area_weight * np.dstack([tdata1.reshape(-1,1), tdata2.reshape(-1,1)]).copy("C"))).astype(np.int32)

# create potts pairwise
        #pairwiseAlpha = -10
        pairwise = -(np.eye(2)-1)
        pairwise = (self.segparams['pairwise_alpha'] * pairwise).astype(np.int32)
        #pairwise = np.array([[0,30],[30,0]]).astype(np.int32)
        #print pairwise

        self.iparams = {}

# use the gerneral graph algorithm
# first, we construct the grid graph
        inds = np.arange(data.size).reshape(data.shape)
        if self.segparams['use_boundary_penalties']:
#  některém testu  organ semgmentation dosahují unaries -15. což je podiné
# stačí yhodit print před if a je to idět
            print "unaries %.3g , %.3g" % (np.max(unariesalt), np.min(unariesalt))
            bpw = self.segparams['boundary_penalties_weight']
            sigma = self.segparams['boundary_penalties_sigma']
            bpa = self.boundary_penalties_array(axis=2, sigma=sigma)
            #id1=inds[:, :, :-1].ravel()
            edgx = np.c_[
                    inds[:, :, :-1].ravel(),
                    inds[:, :, 1:].ravel(),
                    #cc * np.ones(id1.shape)]
                    bpw* bpa[:,:,1:].ravel()]

            bpa = self.boundary_penalties_array(axis=1, sigma=sigma)
            #id1 =inds[:, 1:, :].ravel()
            edgy = np.c_[
                    inds[:, :-1, :].ravel(),
                    inds[:, 1:, :].ravel(),
                    #cc * np.ones(id1.shape)]
                    bpw* bpa[:, 1:,:].ravel()]

            bpa = self.boundary_penalties_array(axis=0, sigma=sigma)
            #id1 = inds[1:, :, :].ravel()
            edgz = np.c_[
                    inds[:-1, :, :].ravel(),
                    inds[1:, :, :].ravel(),
                    #cc * np.ones(id1.shape)]
                    bpw * bpa[1:,:,:].ravel()]
        else:

            edgx = np.c_[inds[:, :, :-1].ravel(), inds[:, :, 1:].ravel()]
            edgy = np.c_[inds[:, :-1, :].ravel(), inds[:, 1:, :].ravel()]
            edgz = np.c_[inds[:-1, :, :].ravel(), inds[1:, :, :].ravel()]

        #import pdb; pdb.set_trace()
        edges = np.vstack([edgx, edgy, edgz]).astype(np.int32)

# edges - seznam indexu hran, kteres spolu sousedi

# we flatten the unaries
        #result_graph = cut_from_graph(edges, unaries.reshape(-1, 2), pairwise)
        result_graph = cut_from_graph(edges, unariesalt.reshape(-1,2), pairwise)

        #print "unaries %.3g , %.3g" % (np.max(unariesalt), np.min(unariesalt))
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
                          variable_names=['data', 'voxelsize_mm'])

    igc = ImageGraphCut(dataraw['data'], voxelsize=dataraw['voxelsize_mm'])
    igc.interactivity()

    logger.debug(igc.segmentation.shape)

if __name__ == "__main__":
    main()
