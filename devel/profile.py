# cmd> snakeviz hi2lo.profile

import numpy as np
import scipy
from imcut import pycut
import cProfile
# import io3d

def make_data(sz=32, offset=0, sigma=80):
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

img, seg, seeds = make_data(64, 20)
segparams = {
    # 'method':'graphcut',
    # "method": "multiscale_graphcut",
    # "method": "hi2lo",
    "method": "lo2hi",
    "use_boundary_penalties": False,
    "boundary_dilatation_distance": 2,
    "boundary_penalties_weight": 1,
    "block_size": 8,
    "tile_zoom_constant": 1,
}
gc = pycut.ImageGraphCut(img, segparams=segparams)
gc.set_seeds(seeds)
gc.run()
# cProfile.run("gc.run()")
# import sed3
# ed = sed3.sed3(gc.segmentation==0, contour=seg)
# ed.show()

# self.assertLess(
#     np.sum(
#         np.abs((gc.segmentation == 0).astype(np.int8) - seg.astype(np.int8))
#     ),
#     600,
# )
