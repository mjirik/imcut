#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
# import os.path as op

logger = logging.getLogger(__name__)
import numpy as np
import scipy


def resize_to_shape(data, shape, zoom=None, mode='nearest', order=0):
    """
    Function resize input data to specific shape.
    :param data: input 3d array-like data
    :param shape: shape of output data
    :param zoom: zoom is used for back compatibility
    :mode: default is 'nearest'
    """
    # @TODO remove old code in except part
    # TODO use function from library in future

    try:
        # rint 'pred vyjimkou'
        # aise Exception ('test without skimage')
        # rint 'za vyjimkou'
        import skimage
        import skimage.transform
        # Now we need reshape  seeds and segmentation to original size

        segm_orig_scale = skimage.transform.resize(
            data, shape, order=0,
            preserve_range=True
        )

        segmentation = segm_orig_scale
        logger.debug('resize to orig with skimage')
    except:
        if zoom is None:
            zoom = shape / np.asarray(data.shape).astype(np.double)
        segmentation = resize_to_shape_with_zoom(
            data,
            zoom=zoom,
            mode=mode,
            order=order
        )

    return segmentation


def resize_to_shape_with_zoom(data, shape, zoom, mode='nearest', order=0):
    import scipy
    import scipy.ndimage
    dtype = data.dtype

    segm_orig_scale = scipy.ndimage.zoom(
        data,
        1.0 / zoom,
        mode=mode,
        order=order
    ).astype(dtype)
    logger.debug('resize to orig with scipy.ndimage')

    # @TODO odstranit hack pro oříznutí na stejnou velikost
    # v podstatě je to vyřešeno, ale nechalo by se to dělat elegantněji v zoom
    # tam je bohužel patrně bug
    # rint 'd3d ', self.data3d.shape
    # rint 's orig scale shape ', segm_orig_scale.shape
    shp = [
        np.min([segm_orig_scale.shape[0], shape[0]]),
        np.min([segm_orig_scale.shape[1], shape[1]]),
        np.min([segm_orig_scale.shape[2], shape[2]]),
    ]
    # elf.data3d = self.data3d[0:shp[0], 0:shp[1], 0:shp[2]]
    # mport ipdb; ipdb.set_trace() # BREAKPOINT

    segmentation = np.zeros(shape, dtype=dtype)
    segmentation[
    0:shp[0],
    0:shp[1],
    0:shp[2]] = segm_orig_scale[0:shp[0], 0:shp[1], 0:shp[2]]

    del segm_orig_scale
    return segmentation


def seed_zoom(seeds, zoom):
    """
    Smart zoom for sparse matrix. If there is resize to bigger resolution
    thin line of label could be lost. This function prefers labels larger
    then zero. If there is only one small voxel in larger volume with zeros
    it is selected.
    """
    # import scipy
    # loseeds=seeds
    labels = np.unique(seeds)
    # remove first label - 0
    labels = np.delete(labels, 0)
    # @TODO smart interpolation for seeds in one block
    #        loseeds = scipy.ndimage.interpolation.zoom(
    #            seeds, zoom, order=0)
    loshape = np.ceil(np.array(seeds.shape) * 1.0 / zoom).astype(np.int)
    loseeds = np.zeros(loshape, dtype=np.int8)
    loseeds = loseeds.astype(np.int8)
    for label in labels:
        a, b, c = np.where(seeds == label)
        loa = np.round(a // zoom)
        lob = np.round(b // zoom)
        loc = np.round(c // zoom)
        # loseeds = np.zeros(loshape)

        loseeds[loa, lob, loc] += label
        # this is to detect conflict seeds
        loseeds[loseeds > label] = 100

    # remove conflict seeds
    loseeds[loseeds > 99] = 0

    # import py3DSeedEditor
    # ped = py3DSeedEditor.py3DSeedEditor(loseeds)
    # ped.show()

    return loseeds

def zoom_to_shape(data, shape, dtype=None):
    """
    Zoom data to specific shape.
    """
    import scipy
    import scipy.ndimage
    zoomd = np.array(shape) / np.array(data.shape, dtype=np.double)
    datares = scipy.ndimage.interpolation.zoom(data, zoomd, order=0)

    if datares.shape != shape:
        logger.warning('Zoom with different output shape')
    dataout = np.zeros(shape, dtype=dtype)
    shpmin = np.minimum(dataout.shape, shape)

    dataout[:shpmin[0], :shpmin[1], :shpmin[2]] = datares[
                                                  :shpmin[0], :shpmin[1], :shpmin[2]]
    return datares


def getPriorityObjects(data, nObj=1, seeds=None, debug=False):
    """

    Vraceni N nejvetsich objektu.
        input:
            data - data, ve kterych chceme zachovat pouze nejvetsi objekty
            nObj - pocet nejvetsich objektu k vraceni
            seeds - dvourozmerne pole s umistenim pixelu, ktere chce uzivatel
                vratit (odpovidaji matici "data")

        returns:
            data s nejvetsimi objekty

    """

    # Oznaceni dat.
    # labels - oznacena data.
    # length - pocet rozdilnych oznaceni.
    dataLabels, length = scipy.ndimage.label(data)

    logger.info('Olabelovano oblasti: ' + str(length))

    if debug:
        logger.debug('data labels: ' + str(dataLabels))

    # Uzivatel si nevybral specificke objekty.
    if (seeds == None):

        logger.info('Vraceni bez seedu')
        logger.debug('Objekty: ' + str(nObj))

        # Zjisteni nejvetsich objektu.
        arrayLabelsSum, arrayLabels = areaIndexes(dataLabels, length)
        # Serazeni labelu podle velikosti oznacenych dat (prvku / ploch).
        arrayLabelsSum, arrayLabels = selectSort(arrayLabelsSum, arrayLabels)

        returning = None
        label = 0
        stop = nObj - 1

        # Budeme postupne prochazet arrayLabels a postupne pridavat jednu
        # oblast za druhou (od te nejvetsi - mimo nuloveho pozadi) dokud
        # nebudeme mit dany pocet objektu (nObj).
        while label <= stop:

            if label >= len(arrayLabels):
                break

            if arrayLabels[label] != 0:
                if returning == None:
                    # "Prvni" iterace
                    returning = data * (dataLabels == arrayLabels[label])
                else:
                    # Jakakoli dalsi iterace
                    returning = returning + data * \
                                (dataLabels == arrayLabels[label])
            else:
                # Musime prodlouzit hledany interval, protoze jsme narazili na
                # nulove pozadi.
                stop = stop + 1

            label = label + 1

            if debug:
                logger.debug(str(label - 1) + ': ' + str(returning))

        if returning == None:
            logger.info(
                'Zadna validni olabelovana data! (DEBUG: returning == None)')

        return returning

    # Uzivatel si vybral specificke objekty (seeds != None).
    else:

        logger.info('Vraceni se seedy')

        # Zalozeni pole pro ulozeni seedu
        arrSeed = []
        # Zjisteni poctu seedu.
        stop = seeds[0].size
        tmpSeed = 0
        dim = np.ndim(dataLabels)
        for index in range(0, stop):
            # Tady se ukladaji labely na mistech, ve kterych kliknul uzivatel.
            if dim == 3:
                # 3D data.
                tmpSeed = dataLabels[
                    seeds[0][index], seeds[1][index], seeds[2][index]]
            elif dim == 2:
                # 2D data.
                tmpSeed = dataLabels[seeds[0][index], seeds[1][index]]

            # Tady opet pocitam s tim, ze oznaceni nulou pripada cerne oblasti
            # (pozadi).
            if tmpSeed != 0:
                # Pokud se nejedna o pozadi (cernou oblast), tak se novy seed
                # ulozi do pole "arrSeed"
                arrSeed.append(tmpSeed)

        # Pokud existuji vhodne labely, vytvori se nova data k vraceni.
        # Pokud ne, vrati se "None" typ. { Deprecated: Pokud ne, vrati se cela
        # nafiltrovana data, ktera do funkce prisla (nedojde k vraceni
        # specifickych objektu). }
        if len(arrSeed) > 0:

            # Zbaveni se duplikatu.
            arrSeed = list(set(arrSeed))
            if debug:
                logger.debug('seed list:' + str(arrSeed))

            logger.info(
                'Ruznych prioritnich objektu k vraceni: ' +
                str(len(arrSeed))
            )

            # Vytvoreni vystupu - postupne pricitani dat prislunych specif.
            # labelu.
            returning = None
            for index in range(0, len(arrSeed)):

                if returning == None:
                    returning = data * (dataLabels == arrSeed[index])
                else:
                    returning = returning + data * \
                                (dataLabels == arrSeed[index])

                if debug:
                    logger.debug((str(index)) + ':' + str(returning))

            return returning

        else:

            logger.info(
                'Zadna validni data k vraceni - zadne prioritni objekty ' +
                'nenalezeny (DEBUG: function getPriorityObjects:' +
                str(len(arrSeed) == 0))
            return None

