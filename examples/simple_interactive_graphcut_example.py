#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import imcut.pycut
import scipy.ndimage


def make_data(sz=32, offset=0, sigma=80):
    seeds = np.zeros([sz, sz, sz], dtype=np.int8)
    seeds[offset + 12, offset + 9:offset + 14, offset + 10] = 1
    print(offset)
    seeds[offset + 20, offset + 18:offset + 21, offset + 12] = 1
    img = np.ones([sz, sz, sz])
    img = img - seeds

    seeds[
    offset + 3:offset + 15,
    offset + 2:offset + 6,
    offset + 27:offset + 29] = 2
    img = scipy.ndimage.morphology.distance_transform_edt(img)
    segm = img < 7
    img = (100 * segm + sigma * np.random.random(img.shape)).astype(np.uint8)
    return img, segm, seeds


def main():
    segparams_ssgc = {
        # 'method':'graphcut',
        'method': 'graphcut',
        'use_boundary_penalties': False,
        'boundary_dilatation_distance': 2,
        'boundary_penalties_weight': 1,
        'modelparams': {
            'type': 'gmmsame',
            "params": {
                "n_components": 2,
            },
            # "return_only_objects_with_seeds": True,
            # 'fv_type': "fv_extern",
            # 'fv_extern': fv_function,
            # 'adaptation': 'original_data',
        }
    }
    img, seg, seeds = make_data(64, 20)
    gc = imcut.pycut.ImageGraphCut(img, segparams=segparams_ssgc)
    gc.set_seeds(seeds)

    # gc.run()
    gc.interactivity()
    print(gc.segmentation)
    err = np.sum(np.abs((gc.segmentation == 0).astype(np.int8) - seg.astype(np.int8)))
    assert(err < 600)


if __name__ == "__main__":
    main()
