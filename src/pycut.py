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

import pygco
#from pygco import cut_from_graph

import sklearn
import sklearn.mixture

# version comparison
from pkg_resources import parse_version


if parse_version(sklearn.__version__) > parse_version('0.10'):
    #new versions
    defaultmodelparams = {'type': 'gmmsame',
                          'params': {'covariance_type': 'full'}}
else:
    defaultmodelparams = {'type': 'gmmsame',
                          'params': {'cvtype': 'full'}}


class Model:
    """ Model for image intensity. Last dimension represent feature vector.
    m = Model()
    m.train(cla, clb)
    X = numpy.random.random([2,3,4])
    # we have data 2x3 with fature vector with 4 fatures
    m.likelihood(X,0)

    modelparams['type']: type of model estimation. Gaussian mixture from EM
    algorithm is implemented as 'gmmsame'. Gaussian kernel density estimation
    is implemented as 'gaussian_kde'. General kernel estimation ('kernel')
    is from scipy version 0.14 and it is not tested.
    """
    def __init__ (self, nObjects=2, modelparams=defaultmodelparams):

        self.mdl = {}
        self.modelparams = modelparams

    def train(self, clx, cl):
        """ Train clas number cl with data clx.

        clx: data, 2d matrix
        cl: label, integer
        """

        if self.modelparams['type'] == 'gmmsame':
            if len(clx.shape) == 1:
     #  je to jen jednorozměrný vektor, tak je potřeba to převést na 2d matici
                clx = clx.reshape(-1, 1)
            gmmparams = self.modelparams['params']
            self.mdl[cl] = sklearn.mixture.GMM(**gmmparams)
            self.mdl[cl].fit(clx)

        elif self.modelparams['type'] == 'kernel':
            from sklearn.neighbors.kde import KernelDensity
            kernelmodelparams = {'kernel': 'gaussian', 'bandwidth': 0.2}
            self.mdl[cl] = KernelDensity(**kernelmodelparams).fit(clx)
        elif self.modelparams['type'] == 'gaussian_kde':
           # print clx
            import scipy.stats
           # from PyQt4.QtCore import pyqtRemoveInputHook
           # pyqtRemoveInputHook()
           # import ipdb; ipdb.set_trace() # BREAKPOINT
            self.mdl[cl] = scipy.stats.gaussian_kde(clx)
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
            outsha = sha
        else:
            xr = x.reshape(-1, sha[-1])
            outsha = sha[:-1]
            #from PyQt4.QtCore import pyqtRemoveInputHook
            #pyqtRemoveInputHook()
            #import ipdb; ipdb.set_trace() # BREAKPOINT
        if self.modelparams['type'] == 'gmmsame':

            px = self.mdl[cl].score(xr)

#todo ošetřit více dimenzionální fv
            px = px.reshape(outsha)
        elif self.modelparams['type'] == 'kernel':
            px = self.mdl[cl].score_samples(xr)
        elif self.modelparams['type'] == 'gaussian_kde':
            # print x
# np.log because it is likelihood
            px = np.log(self.mdl[cl](xr.reshape(-1)))
            px = px.reshape(outsha)
            #from PyQt4.QtCore import pyqtRemoveInputHook
            #pyqtRemoveInputHook()
            #import ipdb; ipdb.set_trace() # BREAKPOINT
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
                 voxelsize=None,
                 debug_images=False,
                 volume_unit='mm3'
                 ):
        logger.debug('modelparams: ' + str(modelparams) + ' segparams: ' +
                     str(segparams) + " voxelsize: " + str(voxelsize) +
                     " debug_images: " + str(debug_images))

# default values                              use_boundary_penalties
        #self.segparams = {'pairwiseAlpha':10, 'use_boundary_penalties':False}
        self.segparams = {
            'type': 'graphcut',
            'pairwise_alpha': 20,
            'use_boundary_penalties': False,
            'boundary_penalties_sigma': 200,
            'boundary_penalties_weight': 30
        }
        self.segparams.update(segparams)

        self.img = img
        self.tdata = {}
        self.segmentation = None
        self.imgshape = img.shape
        self.modelparams = modelparams
        #self.segparams = segparams
        self.seeds = np.zeros(self.img.shape, dtype=np.int8)
        self.debug_images = debug_images
        self.volume_unit = volume_unit

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
        #import sys
        #print "logger ", logging.getLogger().getEffectiveLevel()
        #from guppy import hpy
        #h = hpy()
        #print h.heap()
        #import pdb

        #logger.debug("obj gc   " + str(sys.getsizeof(self)))

        if self.segparams['type'] in ('graphcut'):

            self.seeds = pyed.getSeeds()
            self.voxels1 = pyed.getSeedsVal(1)
            self.voxels2 = pyed.getSeedsVal(2)

            self.make_gc()

            pyed.setContours(1 - self.segmentation.astype(np.int8))
        elif self.segparams['type'] in ('multiscale_gc'):
            self.__multiscale_gc(pyed)

        try:
            import audiosupport
            audiosupport.beep()
        except:
            print("cannot open audiosupport")

    def __seed_zoom(self, seeds, zoom):
        """
        Smart zoom for sparse matrix. If there is resize to bigger resolution
        thin line of label could be lost. This function prefers labels larger
        then zero. If there is only one small voxel in larger volume with zeros
        it is selected.
        """
        #import scipy
        # loseeds=seeds
        labels = np.unique(seeds)
# remove first label - 0
        labels = np.delete(labels, 0)
        print 'labels', labels
# @TODO smart interpolation for seeds in one block
#        loseeds = scipy.ndimage.interpolation.zoom(
#            seeds, zoom, order=0)
        loshape = np.ceil(np.array(seeds.shape) * 1.0 / zoom)
        loseeds = np.zeros(loshape, dtype=np.int8)
        print 'loseeds.shape ', loseeds.shape
        loseeds = loseeds.astype(np.int8)
        for label in labels:
            a, b, c = np.where(seeds == label)
            loa = np.round(a / zoom)
            lob = np.round(b / zoom)
            loc = np.round(c / zoom)
            #loseeds = np.zeros(loshape)

            loseeds[loa, lob, loc] = label


       # import py3DSeedEditor
       # ped = py3DSeedEditor.py3DSeedEditor(loseeds)
       # ped.show()

        #import ipdb; ipdb.set_trace() # BREAKPOINT
        return loseeds


    def __multiscale_gc(self, pyed):
        import py3DSeedEditor as ped

        from PyQt4.QtCore import pyqtRemoveInputHook
        pyqtRemoveInputHook()
        import scipy
        import scipy.ndimage
        zoom = 8 #0.125 #self.segparams['scale']
        loseeds = pyed.getSeeds()
        print np.unique(loseeds)
        loseeds = self.__seed_zoom(loseeds, zoom)
        print np.unique(loseeds)

        self.seeds = loseeds
        self.voxels1 = pyed.getSeedsVal(1)
        self.voxels2 = pyed.getSeedsVal(2)

        img_orig = self.img

        self.img = scipy.ndimage.interpolation.zoom(img_orig, 1.0/zoom, order=0)

        self.make_gc()
        print 'segmentation'
        print np.max(self.segmentation)
        print np.min(self.segmentation)
        seg = 1 - self.segmentation.astype(np.int8)

        segl = scipy.ndimage.filters.laplace(seg, mode='constant')
        print np.max(segl)
        print np.min(segl)
        segl[segl!=0] = 1
        print np.max(segl)
        print np.min(segl)
        seg = scipy.ndimage.morphology.binary_dilation(
            seg
            #np.ones([3,3,3])
        )
        print seg.shape
        segz = scipy.ndimage.interpolation.zoom(seg.astype('float'), zoom,
                                                order=0).astype('int8')
        #segz [segz > 0.1] = 1
        #segz.astype('int8')
        #import pdb; pdb.set_trace() # BREAKPOINT
# @todo back resize
        segzz = np.zeros(img_orig.shape, dtype='int8')
        segzz [:segz.shape[0],:segz.shape[1],:segz.shape[2]]=segz
        pyed.img = segzz * 100
        msinds = self.__multiscale_indexes(seg, img_orig.shape, zoom, pyed)
        print 'msinds'
        pd = ped.py3DSeedEditor(msinds)
        pd.show()
        import pdb; pdb.set_trace() # BREAKPOINT

        # intensity values for indexes
        # @TODO compute average values for low resolution

        #import pdb; pdb.set_trace() # BREAKPOINT
        #pyed.setContours(seg)

    def __ordered_values_by_indexes(self, data, inds):
        """
        Return values (intensities) by indexes.

        Used for multiscale graph cut.
        data = [[0 1 1],
                [0 2 2],
                [0 2 2]]

        inds = [[0 1 2],
                [3 4 4],
                [5 4 4]]

        """


    def __relabel(self, data):
        """
        Makes relabeling of data if there are unused values.
        """
        palette, index = np.unique(data,return_inverse=True)
        data = index.reshape(data.shape)
# realy slow solution
#        unq = np.unique(data)
#        actual_label = 0
#        for lab in unq:
#            data[data == lab] = actual_label
#            actual_label += 1

        # one another solution probably slower
        # arr = data
        # data = (np.digitize(arr.reshape(-1,),np.unique(arr))-1).reshape(arr.shape)

        return data


    def __multiscale_indexes(self, mask, orig_shape, zoom, pyed):
        """
        Function computes multiscale indexes of ndarray.
        mask: Says where is original resolution (0) and where is small
        resolution (1). Mask is in small resolution.

        orig_shape: Original shape of input data.
        zoom: Usually number greater then 1
        """

        inds_small = np.arange(mask.size).reshape(mask.shape)
        inds_small_in_orig = self.__zoom_to_shape(inds_small, zoom,
                                                  orig_shape, dtype=np.int8)
        inds_orig = np.arange(np.prod(orig_shape)).reshape(orig_shape)

        mask_orig = self.__zoom_to_shape(mask, zoom,
                                         orig_shape, dtype=np.int8)

        inds_orig += np.max(inds_small_in_orig) + 1
        #print 'indexes'
        #import py3DSeedEditor as ped
        #import pdb; pdb.set_trace() # BREAKPOINT
        inds_small_in_orig[mask_orig == True] = inds_orig[mask_orig == True]
        inds = inds_small_in_orig
        print np.max(inds)
        print np.min(inds)
        inds = self.__relabel(inds)
        print np.max(inds)
        print np.min(inds)
        #inds_orig[mask_orig==True] = 0
        #inds_small_in_orig[mask_orig==False] = 0
        #inds = (inds_orig + np.max(inds_small_in_orig) + 1) + inds_small_in_orig

        return inds

        pass

    def __zoom_to_shape(self, data, zoom, shape, dtype=None):
        """
        Zoom data to specific shape.
        """
        import scipy
        import scipy.ndimage
        datares = scipy.ndimage.interpolation.zoom(data, zoom, order=0)
        dataout = np.zeros(shape, dtype=dtype)
        dataout [:shape[0], :shape[1], :shape[2]] = datares
        return datares





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
                            seeds=self.seeds,
                            volume_unit=self.volume_unit
                            )

        pyed.changeC(window_c)
        pyed.changeW(window_w)

        qt_app.exec_()

    def set_seeds(self, seeds):
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
    def __create_multiscale_tlinks(self, ):
        pass

    def __similarity_for_tlinks_obj_bgr(self, data, voxels1, voxels2,
                             seeds, otherfeatures=None):
        """
        Compute edge values for graph cut tlinks based on image intensity
        and texture.
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
        return tdata1, tdata2


    def __create_tlinks(self, data, voxels1, voxels2, seeds,
                                          area_weight, hard_constraints):
        tdata1, tdata2 = self.__similarity_for_tlinks_obj_bgr(data, voxels1,
                                                   voxels2, seeds)
        if self.debug_images:
### Show model parameters
            import matplotlib.pyplot as plt
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.imshow(tdata1[5, :, :])
            print 'max ', np.max(tdata1), 'min ', np.min(tdata1)

            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.imshow(tdata2[5, :, :])
            print 'max ', np.max(tdata2), 'min ', np.min(tdata2)

            fig = plt.figure()
            ax = fig.add_subplot(111)
            hstx = np.linspace(-1000, 1000, 400)
            ax.plot(hstx, mdl.likelihood(hstx, 1))
            ax.plot(hstx, mdl.likelihood(hstx, 2))

            plt.show()

        if hard_constraints:
            #pdb.set_trace();
            if (type(seeds)=='bool'):
                raise Exception (
                    'Seeds variable  not set',
                    'There is need set seed if you want use hard constraints')
            tdata1, tdata2 = self.set_hard_hard_constraints(tdata1,
                                                            tdata2,
                                                            seeds)

        unariesalt = (0 + (area_weight *
                           np.dstack([tdata1.reshape(-1, 1),
                                      tdata2.reshape(-1, 1)]).copy("C"))
                      ).astype(np.int32)
        return unariesalt

    def __create_nlinks(self, data):
# @TODO copy into __create_graph_function
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
        return edges

    def set_data(self, data, voxels1, voxels2,
                 seeds=False,
                 hard_constraints=True,
                 area_weight=1):
        """
        Setting of data.
        You need set seeds if you want use hard_constraints.
        """
        #from PyQt4.QtCore import pyqtRemoveInputHook
        #pyqtRemoveInputHook()
        #import pdb; pdb.set_trace() # BREAKPOINT

        unariesalt = self.__create_tlinks(data, voxels1, voxels2, seeds,
                                          area_weight, hard_constraints)
# create potts pairwise
        #pairwiseAlpha = -10
        pairwise = -(np.eye(2)-1)
        pairwise = (self.segparams['pairwise_alpha'] * pairwise).astype(np.int32)
        #pairwise = np.array([[0,30],[30,0]]).astype(np.int32)
        #print pairwise

        self.iparams = {}

        nlinks = self.__create_nlinks(data)

# edges - seznam indexu hran, kteres spolu sousedi

# we flatten the unaries
        #result_graph = cut_from_graph(nlinks, unaries.reshape(-1, 2), pairwise)
        result_graph = pygco.cut_from_graph(nlinks, unariesalt.reshape(-1,2), pairwise)

# probably not necessary
#        del nlinks
#        del unariesalt

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
#@profile
def main():
    logger = logging.getLogger()
    logger.setLevel(logging.WARNING)
    logging.basicConfig(format='%(message)s')
    ch = logging.StreamHandler()
    formatter = logging.Formatter("%(levelname)-5s [%(module)s:%(funcName)s:%(lineno)d] %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    parser = OptionParser(description='Organ segmentation')
    parser.add_option('-d','--debug', action='store_true',
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

    debug_images = False

    if options.debug:
        logger.setLevel(logging.DEBUG)
        debug_images = True
        #print DEBUG
        #DEBUG = True

    # if options.tests:
    #     sys.argv[1:]=[]
    #     unittest.main()

    if options.in_filename is None:
        raise IOError('No input data!')

    else:
        dataraw = loadmat(options.in_filename,
                          variable_names=['data', 'voxelsize_mm'])
    #import pdb; pdb.set_trace() # BREAKPOINT

    logger.debug('\nvoxelsize_mm ' + dataraw['voxelsize_mm'].__str__())

    if sys.platform == 'win32':
# hack, on windows is voxelsize read as 2D array like [[1, 0.5, 0.5]]
        dataraw['voxelsize_mm'] = dataraw['voxelsize_mm'][0]


    igc = ImageGraphCut(dataraw['data'], voxelsize=dataraw['voxelsize_mm'],
                        debug_images=debug_images
#                        , modelparams={'type':'gaussian_kde', 'params':[]}
                        , segparams = {'type':'multiscale_gc'} # multiscale gc
                        )
    igc.interactivity()

    logger.debug(igc.segmentation.shape)

if __name__ == "__main__":
    main()
