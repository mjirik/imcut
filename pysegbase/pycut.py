#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Organ segmentation

Example:

$ pycat -f head.mat -o brain.mat
"""

from __future__ import absolute_import, division, print_function

# import unittest
# from optparse import OptionParser
import argparse
import sys
import logging
# import os.path as op

logger = logging.getLogger(__name__)

from scipy.io import loadmat
import numpy as np
import copy
import pygco
# from pygco import cut_from_graph

# from . import models
from .image_manipulation import seed_zoom, zoom_to_shape, resize_to_shape, resize_to_shape_with_zoom, getPriorityObjects

from .models import Model, Model3D, defaultmodelparams, methods



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

    def __init__(self,
                 img,
                 modelparams={},
                 segparams={},
                 voxelsize=[1,1,1],
                 debug_images=False,
                 volume_unit='mm3',
                 interactivity_loop_finish_fcn=None,
                 ):
        """

        Args:
            img: input data
            modelparams: parameters of model
            segparams: segmentation parameters
                use_apriori_if_available - set self.apriori to ndimage with same shape as img
                apriori_gamma: influence of apriory information. 0 means no influence, 1.0 is 100% use of
                apriori information
            voxelsize: size of voxel
            debug_images: use to show debug images with matplotlib
            volume_unit: define string of volume unit. Default is "mm3"

        Returns:

        """

        logger.debug('modelparams: ' + str(modelparams) + ' segparams: ' +
                     str(segparams) + " voxelsize: " + str(voxelsize) +
                     " debug_images: " + str(debug_images))

        # default values                              use_boundary_penalties
        # self.segparams = {'pairwiseAlpha':10, 'use_boundary_penalties':False}
        self.segparams = {
            'method': 'graphcut',
            'pairwise_alpha': 20,
            'use_boundary_penalties': False,
            'boundary_penalties_sigma': 200,
            'boundary_penalties_weight': 30,
            'return_only_object_with_seeds': False,
            'use_old_similarity': True,  # New similarity not implemented @TODO
            'use_extra_features_for_training': False,
            'use_apriori_if_available': True,
            'apriori_gamma': 0.1,
        }
        if 'modelparams' in segparams.keys():
            modelparams = segparams['modelparams']
        self.segparams.update(segparams)

        self.img = img
        self.tdata = {}
        self.segmentation = None
        self.imgshape = img.shape
        self.modelparams = defaultmodelparams.copy()
        self.modelparams.update(modelparams)
        # self.segparams = segparams
        self.seeds = np.zeros(self.img.shape, dtype=np.int8)
        self.debug_images = debug_images
        self.volume_unit = volume_unit

        self.voxelsize = np.asarray(voxelsize)
        if voxelsize is not None:
            self.voxel_volume = np.prod(voxelsize)

        else:
            self.voxel_volume = None

        self.interactivity_counter = 0
        self.stats = {
            'tlinks shape': [],
            'nlinks shape': []
        }
        self.mdl = Model(modelparams=self.modelparams)
        self.apriori = None
        self.interactivity_loop_finish_funcion=interactivity_loop_finish_fcn

    def interactivity_loop(self, pyed):
        # @TODO stálo by za to, přehodit tlačítka na myši. Levé má teď
        # jedničku, pravé dvojku. Pravým však zpravidla označujeme pozadí a tak
        # nám vyjde popředí jako nula a pozadí jako jednička.
        # Tím také dopadne jinak interaktivní a neinteraktivní varianta.
        # import sys
        # print "logger ", logging.getLogger().getEffectiveLevel()
        # from guppy import hpy
        # h = hpy()
        # print h.heap()
        # import pdb

        # logger.debug("obj gc   " + str(sys.getsizeof(self)))

        self.set_seeds(pyed.getSeeds())
        self.run()
        pyed.setContours(1 - self.segmentation.astype(np.int8))


        # if self.interactivity_loop_finish_funcion is None:
        #     # TODO remove this statement after lisa package update (12.5.2018)
        #     from lisa import audiosupport
        #     self.interactivity_loop_finish_funcion = audiosupport.beep

        if self.interactivity_loop_finish_funcion is not None:
            self.interactivity_loop_finish_funcion()
        # try:
        # except:
        #     print("cannot open audiosupport")

        self.interactivity_counter += 1
        logger.debug('interactivity counter: ' +
                     str(self.interactivity_counter))

    def __uniform_npenalty_fcn(self, orig_shape):
        return np.ones(orig_shape, dtype=np.int8)

    def __ms_npenalty_fcn(self, axis, mask, ms_zoom, orig_shape):
        """
        :param axis:
        :param mask: 3d ndarray with ones where is finner resolution

        Neighboorhood penalty between small pixels should be smaller then in
        bigger tiles. This is the way how to set it.

        """
        # import scipy.ndimage.filters as scf
        # Váha velkého pixelu je nastavena na délku jeho úhlopříčky
        # TODO remove TILE_ZOOM_CONSTANT
        # TILE_ZOOM_CONSTANT = self.segparams['block_size'] * 2**0.5
        TILE_ZOOM_CONSTANT = 0.25

        # ms_zoom = ms_zoom * TILE_ZOOM_CONSTANT
        # # for axis in range(0,dim):
        # # filtered = scf.prewitt(self.img, axis=axis)
        maskz = zoom_to_shape(mask, orig_shape)
        # maskz = 1 - maskz.astype(np.int8)
        # maskz = (maskz * (ms_zoom - 1)) + 1


        maskz_new = np.zeros(orig_shape, dtype=np.int16)
        maskz_new[maskz == 0] = ms_zoom * self.segparams['tile_zoom_constant']
        maskz_new[maskz == 1] = 1
        # import sed3
        # ed = sed3.sed3(maskz_new)
        # import ipdb; ipdb.set_trace() #  noqa BREAKPOINT

        return maskz_new

    def __general_gc(self):
        pass

    def __multiscale_gc(self):  # , pyed):
        """
        In first step is performed normal GC.
        Second step construct finer grid on edges of segmentation from first
        step.
        There is no option for use without `use_boundary_penalties`
        """
        deb = False
        # deb = True
        # import py3DSeedEditor as ped
        import time
        start = time.time()

        from PyQt4.QtCore import pyqtRemoveInputHook
        pyqtRemoveInputHook()
        import scipy
        import scipy.ndimage
        logger.debug('performing multiscale_gc')
        # default parameters
        # TODO segparams_lo and segparams_hi je tam asi zbytecně
        sparams_lo = {
            'boundary_dilatation_distance': 2,
            'block_size': 6,
            'use_boundary_penalties': True,
            'boundary_penalties_weight': 1,
            'tile_zoom_constant': 1
        }

        sparams_lo.update(self.segparams)
        sparams_hi = copy.copy(sparams_lo)
        # sparams_lo['boundary_penalties_weight'] = (
        #         sparams_lo['boundary_penalties_weight'] * 
        #         sparams_lo['block_size'])
        self.segparams = sparams_lo

        self.stats["t1"] = (time.time() - start)
        # step 1:  low res GC
        hiseeds = self.seeds
        # ms_zoom = 4  # 0.125 #self.segparams['scale']
        ms_zoom = self.segparams['block_size']
        # loseeds = pyed.getSeeds()
        # logger.debug("msc " + str(np.unique(hiseeds)))
        loseeds = seed_zoom(hiseeds, ms_zoom)

        area_weight = 1
        hard_constraints = True

        self.seeds = loseeds

        modelparams_hi = self.modelparams.copy()
        # feature vector will be computed from selected voxels
        self.modelparams['use_extra_features_for_training'] = True

        # TODO what with voxels? It is used from here
        # hiseeds and hiimage is used to create intensity model
        self.voxels1 = self.img[hiseeds == 1].reshape(-1, 1)
        self.voxels2 = self.img[hiseeds == 2].reshape(-1, 1)
        # this is how to compute with loseeds resolution but in wrong way
        # self.voxels1 = self.img[self.seeds == 1]
        # self.voxels2 = self.img[self.seeds == 2]

        # self.voxels1 = pyed.getSeedsVal(1)
        # self.voxels2 = pyed.getSeedsVal(2)

        img_orig = self.img

        # TODO this should be done with resize_to_shape_whith_zoom
        zoom = np.asarray(loseeds.shape).astype(np.float) / img_orig.shape
        self.img = scipy.ndimage.interpolation.zoom(img_orig,
                                                    zoom,
                                                    order=0)
        voxelsize_orig = self.voxelsize
        logger.debug("zoom " + str(zoom))
        logger.debug("vs" + str(self.voxelsize))
        self.voxelsize = self.voxelsize * zoom

        # self.img = resize_to_shape_with_zoom(img_orig, loseeds.shape, 1.0 / ms_zoom, order=0)

        self.make_gc()
        logger.debug(
            'segmentation - max: %d min: %d' % (
                np.max(self.segmentation),
                np.min(self.segmentation)
            )
        )

        seg = 1 - self.segmentation.astype(np.int8)
        # in seg is now stored low resolution segmentation
        # back to normal parameters
        self.modelparams = modelparams_hi
        self.stats["t2"] = (time.time() - start)
        self.voxelsize = voxelsize_orig
        # step 2: discontinuity localization
        # self.segparams = sparams_hi
        segl = scipy.ndimage.filters.laplace(seg, mode='constant')
        logger.debug(str(np.max(segl)))
        logger.debug(str(np.min(segl)))
        segl[segl != 0] = 1
        logger.debug(str(np.max(segl)))
        logger.debug(str(np.min(segl)))
        # scipy.ndimage.morphology.distance_transform_edt
        boundary_dilatation_distance = self.segparams[
            'boundary_dilatation_distance']
        seg = scipy.ndimage.morphology.binary_dilation(
            seg,
            np.ones([
                (boundary_dilatation_distance * 2) + 1,
                (boundary_dilatation_distance * 2) + 1,
                (boundary_dilatation_distance * 2) + 1
            ])
        )
        if deb:
            import sed3
            pd = sed3.sed3(seg)  # ), contour=seg)
            pd.show()
        # segzoom = scipy.ndimage.interpolation.zoom(seg.astype('float'), zoom,
        #                                                order=0).astype('int8')
        self.stats["t3"] = (time.time() - start)
        # step 3: indexes of new dual graph
        msinds = self.__multiscale_indexes(seg, img_orig.shape, ms_zoom)
        logger.debug('multiscale inds ' + str(msinds.shape))
        # if deb:
        #     import sed3
        #     pd = sed3.sed3(msinds, contour=seg)
        #     pd.show()

        # intensity values for indexes
        # @TODO compute average values for low resolution
        ms_img = img_orig

        # @TODO __ms_create_nlinks , use __ordered_values_by_indexes
        # import pdb; pdb.set_trace() # BREAKPOINT
        # pyed.setContours(seg)

        # there is need to set correct weights between neighbooring pixels
        # this is not nice hack.
        # @TODO reorganise segparams and create_nlinks function
        self.img = img_orig  # not necessary
        orig_shape = img_orig.shape

        def local_ms_npenalty(x):
            return self.__ms_npenalty_fcn(x, seg, ms_zoom, orig_shape)
            # return self.__uniform_npenalty_fcn(orig_shape)

        # ms_npenalty_fcn = lambda x: self.__ms_npenalty_fcn(x, seg, ms_zoom,
        #                                                    orig_shape)


        self.stats["t4"] = (time.time() - start)
        # here are not unique couples of nodes
        nlinks_not_unique = self.__create_nlinks(
            ms_img,
            msinds,
            # boundary_penalties_fcn=ms_npenalty_fcn
            boundary_penalties_fcn=local_ms_npenalty
        )

        self.stats["t5"] = (time.time() - start)

        # get unique set
        # remove repetitive link from one pixel to another
        nlinks = ms_remove_repetitive_link(nlinks_not_unique)
        # now remove cycle link
        self.stats["t6"] = (time.time() - start)
        nlinks = np.array([line for line in nlinks if line[0] != line[1]])

        self.stats["t7"] = (time.time() - start)
        # import ipdb; ipdb.set_trace() #  noqa BREAKPOINT
        # tlinks - indexes, data_merge
        ms_values_lin = self.__ordered_values_by_indexes(img_orig, msinds)
        seeds = hiseeds
        # seeds = pyed.getSeeds()
        # if deb:
        #     import sed3
        #     se = sed3.sed3(seeds)
        #     se.show()
        ms_seeds_lin = self.__ordered_values_by_indexes(seeds, msinds)
        # logger.debug("unique seeds " + str(np.unique(seeds)))
        # logger.debug("unique seeds " + str(np.unique(ms_seeds_lin)))

        # TODO vyresit voxelsize
        unariesalt = self.__create_tlinks(ms_values_lin,
                                          voxelsize=self.voxelsize,
                                          # self.voxels1, self.voxels2,
                                          seeds=ms_seeds_lin,
                                          area_weight=area_weight,
                                          hard_constraints=hard_constraints)

        self.stats["t8"] = (time.time() - start)
        # create potts pairwise
        # pairwiseAlpha = -10
        pairwise = -(np.eye(2) - 1)
        pairwise = (self.segparams['pairwise_alpha'] * pairwise
                    ).astype(np.int32)

        # print 'data shape ', img_orig.shape
        # print 'nlinks sh ', nlinks.shape
        # print 'tlinks sh ', unariesalt.shape

        # print "cut_from_graph"
        # print "unaries sh ", unariesalt.reshape(-1,2).shape
        # print "nlinks sh",  nlinks.shape
        self.stats["t9"] = (time.time() - start)
        self.stats['tlinks shape'].append(unariesalt.reshape(-1, 2).shape)
        self.stats['nlinks shape'].append(nlinks.shape)
        import time
        start = time.time()
        # Same functionality is in self.seg_data()
        result_graph = pygco.cut_from_graph(
            nlinks,
            unariesalt.reshape(-1, 2),
            pairwise
        )

        elapsed = (time.time() - start)
        self.stats['gc time'] = elapsed

        # probably not necessary
        #        del nlinks
        #        del unariesalt

        # print "unaries %.3g , %.3g" % (np.max(unariesalt),np.min(unariesalt))
        # @TODO get back original data
        # result_labeling = result_graph.reshape(data.shape)
        result_labeling = result_graph[msinds]
        # import py3DSeedEditor
        # ped = py3DSeedEditor.py3DSeedEditor(result_labeling)
        # ped.show()
        self.segmentation = result_labeling

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

        return: [0, 1, 1, 0, 2, 0]

        If the data are not consistent, it will take the maximal value

        """
        # get unique labels and their first indexes
        # lab, linds = np.unique(inds, return_index=True)
        # compute values by indexes
        # values = data.reshape(-1)[linds]

        # alternative slow implementation
        # if there are different data on same index, it will take
        # maximal value
        # lab = np.unique(inds)
        # values = [0]*len(lab)
        # for label in lab:
        #     values[label] = np.max(data[inds == label])
        #
        # values = np.asarray(values)

        # yet another implementation
        values = [None] * (np.max(inds) + 1)

        linear_inds = inds.ravel()
        linear_data = data.ravel()
        for i in range(0, len(linear_inds)):
            # going over all data pixels

            if values[linear_inds[i]] is None:
                # this index is found for first
                values[linear_inds[i]] = linear_data[i]
            elif values[linear_inds[i]] < linear_data[i]:
                # here can be changed maximal or minimal value
                values[linear_inds[i]] = linear_data[i]

        values = np.asarray(values)

        return values

    def __relabel(self, data):
        """  Makes relabeling of data if there are unused values.  """
        palette, index = np.unique(data, return_inverse=True)
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

    def __multiscale_indexes(self, mask, orig_shape, zoom):
        """
        Function computes multiscale indexes of ndarray.

        mask: Says where is original resolution (0) and where is small
        resolution (1). Mask is in small resolution.

        orig_shape: Original shape of input data.
        zoom: Usually number greater then 1

        result = [[0 1 2],
                  [3 4 4],
                  [5 4 4]]
        """

        mask_orig = zoom_to_shape(mask, orig_shape, dtype=np.int8)

        inds_small = np.arange(mask.size).reshape(mask.shape)
        inds_small_in_orig = zoom_to_shape(inds_small,
                                           orig_shape, dtype=np.int8)
        inds_orig = np.arange(np.prod(orig_shape)).reshape(orig_shape)

        # inds_orig = inds_orig * mask_orig
        inds_orig += np.max(inds_small_in_orig) + 1
        # print 'indexes'
        # import py3DSeedEditor as ped
        # import pdb; pdb.set_trace() # BREAKPOINT

        #  '==' is not the same as 'is' for numpy.array
        inds_small_in_orig[mask_orig == True] = inds_orig[mask_orig == True]  # noqa
        inds = inds_small_in_orig
        # print np.max(inds)
        # print np.min(inds)
        inds = self.__relabel(inds)
        logger.debug("Maximal index after relabeling: " + str(np.max(inds)))
        logger.debug("Minimal index after relabeling: " + str(np.min(inds)))
        # inds_orig[mask_orig==True] = 0
        # inds_small_in_orig[mask_orig==False] = 0
        # inds = (inds_orig + np.max(inds_small_in_orig) + 1) + inds_small_in_orig

        return inds

    def __merge_indexes_by_mask(self, mask, inds1, inds2):
        """
        Return array of indexes.

        inds1: Array with numbers starting from 0
        inds2: Array with numbers starting from 0
        mask: array of same size as inds1 and inds2 with 0 where should be
            inds1 and 1 where sould be inds2

        To values of inds2 is added maximal value of inds1.

        """
        inds1[mask == 1]

    def interactivity(self, min_val=None, max_val=None, qt_app=None):
        """
        Interactive seed setting with 3d seed editor
        """
        from .seed_editor_qt import QTSeedEditor
        from PyQt4.QtGui import QApplication
        if min_val is None:
            min_val = np.min(self.img)

        if max_val is None:
            max_val = np.max(self.img)

        window_c = ((max_val + min_val) / 2)  # .astype(np.int16)
        window_w = (max_val - min_val)  # .astype(np.int16)

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
        :param seeds: ndarray (0 - nothing, 1 - object, 2 - background,
        3 - object just hard constraints, no model training, 4 - background 
        just hard constraints, no model training)
        """
        if self.img.shape != seeds.shape:
            raise Exception("Seeds must be same size as input image")

        self.seeds = seeds.astype('int8')
        self.voxels1 = self.img[self.seeds == 1]
        self.voxels2 = self.img[self.seeds == 2]

    def run(self):
        if self.segparams['method'] in ('graphcut', 'GC'):
            self.make_gc()
        elif self.segparams['method'] in ('multiscale_graphcut', "multiscale_gc"):
            self.__multiscale_gc()
        else:
            logger.error('Unknown segmentation method: ' + self.segparams['method'])

    def make_gc(self):
        res_segm = self.prepare_data_and_run_computation(
            # self.img,
            #                      self
                                 # self.voxels1, self.voxels2,
             # seeds=self.seeds
        )

        if self.segparams['return_only_object_with_seeds']:
            try:
                # because of negative problem is as 1 segmented background and
                # as 0 is segmented foreground
                # import thresholding_functions
                # newData = thresholding_functions.getPriorityObjects(
                newData = getPriorityObjects(
                    (1 - res_segm),
                    nObj=-1,
                    seeds=(self.seeds == 1).nonzero(),
                    debug=False
                )
                res_segm = 1 - newData
            except:
                import traceback
                logger.warning('Cannot import thresholding_funcions')
                traceback.print_exc()

        self.segmentation = res_segm.astype(np.int8)

    def set_hard_hard_constraints(self, tdata1, tdata2, seeds):
        """
        it works with seed labels:
        0: nothing
        1: object 1 - full seeds
        2: object 2 - full seeds
        3: object 1 - not a training seeds
        4: object 2 - not a training seeds
        """
        seeds_mask = (seeds == 1) | (seeds == 3)
        tdata2[seeds_mask] = np.max(tdata2) + 1
        tdata1[seeds_mask] = 0

        seeds_mask = (seeds == 2) | (seeds == 4)
        tdata1[seeds_mask] = np.max(tdata1) + 1
        tdata2[seeds_mask] = 0

        return tdata1, tdata2

    def boundary_penalties_array(self, axis, sigma=None):

        import scipy.ndimage.filters as scf

        # for axis in range(0,dim):
        filtered = scf.prewitt(self.img, axis=axis)
        if sigma is None:
            sigma2 = np.var(self.img)
        else:
            sigma2 = sigma ** 2

        filtered = np.exp(-np.power(filtered, 2) / (256 * sigma2))

        # srovnán hodnot tak, aby to vycházelo mezi 0 a 100
        # cc = 10
        # filtered = ((filtered - 1)*cc) + 10
        logger.debug(
            'ax %.1g max %.3g min %.3g  avg %.3g' % (
                axis, np.max(filtered), np.min(filtered), np.mean(filtered))
        )
        #
        # @TODO Check why forumla with exp is not stable
        # Oproti Boykov2001b tady nedělím dvojkou. Ta je tam jen proto,
        # aby to slušně vycházelo, takže jsem si jí upravil
        # Originální vzorec je
        # Bpq = exp( - (Ip - Iq)^2 / (2 * \sigma^2) ) * 1 / dist(p,q)
        #        filtered = (-np.power(filtered,2)/(16*sigma))
        # Přičítám tu 256 což je empiricky zjištěná hodnota - aby to dobře vyšlo
        # nedávám to do exponenciely, protože je to numericky nestabilní
        # filtered = filtered + 255 # - np.min(filtered2) + 1e-30
        # Ještě by tady měl a následovat exponenciela, ale s ní je to numericky
        # nestabilní. Netuším proč.
        # if dim >= 1:
        # odecitame od sebe tentyz obrazek
        # df0 = self.img[:-1,:] - self.img[]
        # diffs.insert(0,
        return filtered

    def __show_debug_(self, unariesalt, suptitle=None):
        shape = self.img.shape
        print("unariesalt dtype ", unariesalt.dtype)
        tdata1 = unariesalt[..., 0].reshape(shape)
        tdata2 = unariesalt[..., 1].reshape(shape)
        self.__show_debug_tdata_images(tdata1, tdata2, suptitle=suptitle)
        pass


    def __show_debug_tdata_images(self, tdata1, tdata2, suptitle=None):
        # Show model parameters
        logger.debug('tdata1 shape ' + str(tdata1.shape))
        slice_number = int(tdata1.shape[0] / 2)
        try:
            import matplotlib.pyplot as plt
            fig = plt.figure()
            fig.suptitle(suptitle)
            ax = fig.add_subplot(121)
            ax.imshow(tdata1[slice_number, :, :])

            # fig = plt.figure()
            ax = fig.add_subplot(122)
            ax.imshow(tdata2[slice_number, :, :])

            print('tdata1 max ', np.max(tdata1), ' min ', np.min(tdata1), " dtype ", tdata1.dtype)
            print('tdata2 max ', np.max(tdata2), ' min ', np.min(tdata2), " dtype ", tdata2.dtype)


            # # histogram
            # fig = plt.figure()
            # vx1 = data[seeds==1]
            # vx2 = data[seeds==2]
            # plt.hist([vx1, vx2], 30)

            # plt.hist(voxels2)

            plt.show()
        except:
            import traceback
            print(traceback.format_exc())
            logger.debug('problem with showing debug images')

        try:
            fig = plt.figure()
            fig.suptitle(suptitle)
            ax = fig.add_subplot(121)
            plt.hist(tdata1.flatten())
            ax = fig.add_subplot(122)
            plt.hist(tdata2.flatten())
        except:
            import traceback
            print(traceback.format_exc())

        try:
            fig = plt.figure()
            fig.suptitle(suptitle)
            ax = fig.add_subplot(111)
            hstx = np.linspace(-1000, 1000, 400)
            ax.plot(hstx, np.exp(self.mdl.likelihood_from_image(hstx, self.voxelsize, 1)))
            ax.plot(hstx, np.exp(self.mdl.likelihood_from_image(hstx, self.voxelsize, 2)))
        except:
            import traceback
            print(traceback.format_exc())

    def __similarity_for_tlinks_obj_bgr(self,
                                        data,
                                        voxelsize,
                                        #voxels1, voxels2,

                                        seeds, otherfeatures=None):
        """
        Compute edge values for graph cut tlinks based on image intensity
        and texture.
        """
        # TODO rewrite just for one class and call separatelly for obj and background.

        # TODO rename voxels1 and voxels2
        # voxe1s1 and voxels2 are used only in this function for multiscale graphcut
        # threre can be some

        # Dobře to fungovalo area_weight = 0.05 a cc = 6 a diference se
        # počítaly z :-1


        # self.mdl.trainFromSomething(data, seeds, 1, self.voxels1)
        # self.mdl.trainFromSomething(data, seeds, 2, self.voxels2)
        if self.segparams['use_extra_features_for_training']:
            self.mdl.fit(self.voxels1, 1)
            self.mdl.fit(self.voxels2, 2)
        else:
            self.mdl.fit_from_image(data, voxelsize, seeds, [1, 2]),
        # as we convert to int, we need to multipy to get sensible values

        # There is a need to have small vaues for good fit
        # R(obj) = -ln( Pr (Ip | O) )
        # R(bck) = -ln( Pr (Ip | B) )
        # Boykov2001b
        # ln is computed in likelihood
        tdata1 = (-(self.mdl.likelihood_from_image(data, voxelsize, 1))) * 10
        tdata2 = (-(self.mdl.likelihood_from_image(data, voxelsize, 2))) * 10

        # to spare some memory
        dtype = np.int16
        if np.any(tdata1 > 32760):
            dtype = np.float32
        if np.any(tdata2 > 32760):
            dtype = np.float32

        if self.segparams['use_apriori_if_available'] and self.apriori is not None:
            logger.debug("using apriori information")
            gamma = self.segparams['apriori_gamma']
            a1 = (-np.log(self.apriori * 0.998 + 0.001)) * 10
            a2 = (-np.log(0.999 - (self.apriori * 0.998))) * 10
            # logger.debug('max ' + str(np.max(tdata1)) + ' min ' + str(np.min(tdata1)))
            # logger.debug('max ' + str(np.max(tdata2)) + ' min ' + str(np.min(tdata2)))
            # logger.debug('max ' + str(np.max(a1)) + ' min ' + str(np.min(a1)))
            # logger.debug('max ' + str(np.max(a2)) + ' min ' + str(np.min(a2)))
            tdata1u = (((1 - gamma) * tdata1) + (gamma * a1)).astype(dtype)
            tdata2u = (((1 - gamma) * tdata2) + (gamma * a2)).astype(dtype)
            tdata1 = tdata1u
            tdata2 = tdata2u
            # logger.debug('   max ' + str(np.max(tdata1)) + ' min ' + str(np.min(tdata1)))
            # logger.debug('   max ' + str(np.max(tdata2)) + ' min ' + str(np.min(tdata2)))
            # logger.debug('gamma ' + str(gamma))

            # import sed3
            # ed = sed3.show_slices(tdata1)
            # ed = sed3.show_slices(tdata2)
            del tdata1u
            del tdata2u
            del a1
            del a2

        # if np.any(tdata1 < 0) or np.any(tdata2 <0):
        #     logger.error("Problem with tlinks. Likelihood is < 0")

        # if self.debug_images:
        #     self.__show_debug_tdata_images(tdata1, tdata2, suptitle="likelihood")
        return tdata1, tdata2

    def __limit(self, tdata1, min_limit=0, max_error=10, max_limit=20000):
        # logger.debug('before limit max ' + np.max(tdata1), 'min ' + np.min(tdata1) + " dtype " +  tdata1.dtype)
        tdata1[tdata1 > max_limit] = max_limit
        tdata1[tdata1 < min_limit] = min_limit
        # tdata1 = models.softplus(tdata1, max_error=max_error, keep_dtype=True)
        # replace inf with large finite number
        # tdata1 = np.nan_to_num(tdata1)
        return tdata1

    def __limit_tlinks(self, tdata1, tdata2):
        tdata1 = self.__limit(tdata1)
        tdata2 = self.__limit(tdata2)

        return tdata1, tdata2

    def __create_tlinks(self,
                        data,
                        voxelsize,

                        # voxels1, voxels2,
                        seeds,
                        area_weight, hard_constraints):
        tdata1, tdata2 = self.__similarity_for_tlinks_obj_bgr(
            data,
            voxelsize,
            # voxels1, voxels2,
            seeds
        )

        # logger.debug('tdata1 min %f , max %f' % (tdata1.min(), tdata1.max()))
        # logger.debug('tdata2 min %f , max %f' % (tdata2.min(), tdata2.max()))
        if hard_constraints:
            if (type(seeds) == 'bool'):
                raise Exception(
                    'Seeds variable  not set',
                    'There is need set seed if you want use hard constraints')
            tdata1, tdata2 = self.set_hard_hard_constraints(tdata1,
                                                            tdata2,
                                                            seeds)

        tdata1 = self.__limit(tdata1)
        tdata2 = self.__limit(tdata2)
        unariesalt = (0 + (area_weight *
                           np.dstack([tdata1.reshape(-1, 1),
                                      tdata2.reshape(-1, 1)]).copy("C"))
                      ).astype(np.int32)
        unariesalt = self.__limit(unariesalt)
        # if self.debug_images:
        #     self.__show_debug_(unariesalt, suptitle="after weighing and limitation")
        return unariesalt

    def __create_nlinks(self, data, inds=None, boundary_penalties_fcn=None):
        """
        Compute nlinks grid from data shape information. For boundary penalties
        are data (intensities) values are used.

        ins: Default is None. Used for multiscale GC. This are indexes of
        multiscale pixels. Next example shows one superpixel witn index 2.
        inds = [
            [1 2 2],
            [3 2 2],
            [4 5 6]]

        boundary_penalties_fcn: is function with one argument - axis. It can
            it can be used for setting penalty weights between neighbooring
            pixels.

        """
        # use the gerneral graph algorithm
        # first, we construct the grid graph
        import time
        start = time.time()
        if inds is None:
            inds = np.arange(data.size).reshape(data.shape)
        # if not self.segparams['use_boundary_penalties'] and \
        #         boundary_penalties_fcn is None :
        if boundary_penalties_fcn is None:
            # This is faster for some specific format
            edgx = np.c_[inds[:, :, :-1].ravel(), inds[:, :, 1:].ravel()]
            edgy = np.c_[inds[:, :-1, :].ravel(), inds[:, 1:, :].ravel()]
            edgz = np.c_[inds[:-1, :, :].ravel(), inds[1:, :, :].ravel()]

        else:
            logger.info('use_boundary_penalties')

            bpw = self.segparams['boundary_penalties_weight']

            bpa = boundary_penalties_fcn(2)
            # id1=inds[:, :, :-1].ravel()
            edgx = np.c_[
                inds[:, :, :-1].ravel(),
                inds[:, :, 1:].ravel(),
                # cc * np.ones(id1.shape)
                bpw * bpa[:, :, 1:].ravel()
            ]

            bpa = boundary_penalties_fcn(1)
            # id1 =inds[:, 1:, :].ravel()
            edgy = np.c_[
                inds[:, :-1, :].ravel(),
                inds[:, 1:, :].ravel(),
                # cc * np.ones(id1.shape)]
                bpw * bpa[:, 1:, :].ravel()
            ]

            bpa = boundary_penalties_fcn(0)
            # id1 = inds[1:, :, :].ravel()
            edgz = np.c_[
                inds[:-1, :, :].ravel(),
                inds[1:, :, :].ravel(),
                # cc * np.ones(id1.shape)]
                bpw * bpa[1:, :, :].ravel()
            ]

        # import pdb; pdb.set_trace()
        edges = np.vstack([edgx, edgy, edgz]).astype(np.int32)
        # edges - seznam indexu hran, kteres spolu sousedi\
        elapsed = (time.time() - start)
        self.stats['_create_nlinks time'] = elapsed
        logger.info("__create nlinks time " + str(elapsed))
        return edges

    def prepare_data_and_run_computation(self,
                                         # voxels1, voxels2,
                                         hard_constraints=True,
                                         area_weight=1):
        """
        Setting of data.
        You need set seeds if you want use hard_constraints.
        """
        # from PyQt4.QtCore import pyqtRemoveInputHook
        # pyqtRemoveInputHook()
        # import pdb; pdb.set_trace() # BREAKPOINT

        unariesalt = self.__create_tlinks(self.img,
                                          self.voxelsize,
                                          # voxels1, voxels2,
                                          self.seeds,
                                          area_weight, hard_constraints)
        #  některém testu  organ semgmentation dosahují unaries -15. což je podiné
        # stačí vyhodit print před if a je to vidět
        logger.debug("unaries %.3g , %.3g" % (
            np.max(unariesalt), np.min(unariesalt)))
        # create potts pairwise
        # pairwiseAlpha = -10
        pairwise = -(np.eye(2) - 1)
        pairwise = (self.segparams['pairwise_alpha'] * pairwise
                    ).astype(np.int32)
        # pairwise = np.array([[0,30],[30,0]]).astype(np.int32)
        # print pairwise

        self.iparams = {}

        if self.segparams['use_boundary_penalties']:
            sigma = self.segparams['boundary_penalties_sigma']
            # set boundary penalties function
            # Default are penalties based on intensity differences
            boundary_penalties_fcn = lambda ax: \
                self.boundary_penalties_array(axis=ax, sigma=sigma)
        else:
            boundary_penalties_fcn = None
        nlinks = self.__create_nlinks(self.img,
                                      boundary_penalties_fcn=boundary_penalties_fcn)

        self.stats['tlinks shape'].append(unariesalt.reshape(-1, 2).shape)
        self.stats['nlinks shape'].append(nlinks.shape)
        # we flatten the unaries
        # result_graph = cut_from_graph(nlinks, unaries.reshape(-1, 2),
        # pairwise)
        import time
        start = time.time()
        if self.debug_images:
            self.__show_debug_(unariesalt)
        result_graph = pygco.cut_from_graph(
            nlinks,
            unariesalt.reshape(-1, 2),
            pairwise
        )
        elapsed = (time.time() - start)
        self.stats['gc time'] = elapsed
        result_labeling = result_graph.reshape(self.img.shape)

        return result_labeling

    def save(self, filename):
        self.mdl.save(filename)



def ms_remove_repetitive_link(nlinks_not_unique):
    # nlinks = np.array(
    #     [list(x) for x in set(tuple(x) for x in nlinks_not_unique)]
    # )
    a = nlinks_not_unique
    nlinks = np.unique(a.view(np.dtype((np.void, a.dtype.itemsize * a.shape[1])))).view(a.dtype).reshape(-1, a.shape[1])

    return nlinks



# class Tests(unittest.TestCase):
#     def setUp(self):
#         pass

#     def test_segmentation(self):
#         data_shp = [16,16,16]
#         data = generate_data(data_shp)
#         seeds = np.zeros(data_shp)
# setting background seeds
#         seeds[:,0,0] = 1
#         seeds[6,8:-5,2] = 2
# x[4:-4, 6:-2, 1:-6] = -1

#         igc = ImageGraphCut(data)
# igc.interactivity()
# instead of interacitivity just set seeeds
#         igc.noninteractivity(seeds)

# instead of showing just test results
# igc.show_segmentation()
#         segmentation = igc.segmentation
# Testin some pixels for result
#         self.assertTrue(segmentation[0, 0, -1] == 0)
#         self.assertTrue(segmentation[7, 9, 3] == 1)
#         self.assertTrue(np.sum(segmentation) > 10)
# pdb.set_trace()
# self.assertTrue(True)


# logger.debug(igc.segmentation.shape)

usage = '%prog [options]\n' + __doc__.rstrip()
help = {
    'in_file': 'input *.mat file with "data" field',
    'out_file': 'store the output matrix to the file',
    'debug': 'debug mode',
    'debug_interactivity': "turn on interactive debug mode",
    'test': 'run unit test',
}


# @profile


def main():
    logger = logging.getLogger()
    logger.setLevel(logging.WARNING)
    logging.basicConfig(format='%(message)s')
    ch = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(levelname)-5s [%(module)s:%(funcName)s:%(lineno)d] %(message)s"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # parser = OptionParser(description='Organ segmentation')

    parser = argparse.ArgumentParser(
        description=__doc__
    )
    parser.add_argument('-d', '--debug', action='store_true',
                        help=help['debug'])
    parser.add_argument('-di', '--debug-interactivity', action='store_true',
                        help=help['debug_interactivity'])
    parser.add_argument('-i', '--input-file', action='store',
                        default=None,
                        help=help['in_file'])
    parser.add_argument('-t', '--tests', action='store_true',
                        help=help['test'])
    parser.add_argument('-o', '--outputfile', action='store',
                        dest='out_filename', default='output.mat',
                        help=help['out_file'])
    # (options, args) = parser.parse_args()
    options = parser.parse_args()

    debug_images = False

    if options.debug:
        logger.setLevel(logging.DEBUG)
        # print DEBUG
        # DEBUG = True

    if options.debug_interactivity:
        debug_images = True

    # if options.tests:
    #     sys.argv[1:]=[]
    #     unittest.main()

    if options.input_file is None:
        raise IOError('No input data!')

    else:
        dataraw = loadmat(options.input_file,
                          variable_names=['data', 'voxelsize_mm'])
    # import pdb; pdb.set_trace() # BREAKPOINT

    logger.debug('\nvoxelsize_mm ' + dataraw['voxelsize_mm'].__str__())

    if sys.platform == 'win32':
        # hack, on windows is voxelsize read as 2D array like [[1, 0.5, 0.5]]
        dataraw['voxelsize_mm'] = dataraw['voxelsize_mm'][0]

    igc = ImageGraphCut(dataraw['data'], voxelsize=dataraw['voxelsize_mm'],
                        debug_images=debug_images  # noqa
                        # , modelparams={'type': 'gaussian_kde', 'params': []}
                        # , modelparams={'type':'kernel', 'params':[]}  #noqa not in  old scipy
                        # , modelparams={'type':'gmmsame', 'params':{'cvtype':'full', 'n_components':3}} # noqa 3 components
                        # , segparams={'type': 'multiscale_gc'}  # multisc gc
                        , segparams={'method': 'multiscale_graphcut'}  # multisc gc
                        # , modelparams={'fv_type': 'fv001'}
                        # , modelparams={'type': 'dpgmm', 'params': {'cvtype': 'full', 'n_components': 5, 'alpha': 10}}  # noqa 3 components
                        )
    igc.interactivity()

    logger.debug('igc interactivity countr: ' + str(igc.interactivity_counter))

    logger.debug(igc.segmentation.shape)


if __name__ == "__main__":
    main()
