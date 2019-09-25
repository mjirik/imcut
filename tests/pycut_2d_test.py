#! /usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import pytest

from imcut import pycut

def test_simple_imcut_2d():
    noise_sigma = 50

    img = 80 + np.random.rand(64, 64) * noise_sigma
    img[12:32, 5:25] = img[12:32, 5:25] + 30
    # cca 400 px

    # seeds
    seeds = np.zeros([64, 64], np.int8)
    seeds[13:29, 18:23] = 1
    seeds[4:9, 3:32] = 2
    # [mm]  10 x 10 x 10        # voxelsize_mm = [1, 4, 3]
    voxelsize_mm = [5, 5, 5]


    pycut.defaultmodelparams

    gc = pycut.ImageGraphCut(img, segparams=None)
    gc.set_seeds(seeds)

    gc.run()
    # import matplotlib.pyplot as plt
    # import skimage.color
    # im = skimage.color.label2rgb(gc.segmentation + (seeds * 2), image=img/255., alpha=0.1)
    # plt.imshow(img, cmap="gray")
    # plt.figure()
    # plt.imshow(im)
    # plt.show()
    labels, counts = np.unique(gc.segmentation, return_counts=True)
    assert counts[0] > 300
    assert counts[1] > 300

