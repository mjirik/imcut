import numpy as np
import imcut.pycut as pspc
import matplotlib.pyplot as plt

# create data
data = np.random.rand(30, 30, 30)
data[10:20, 5:15, 3:13] += 1
data = data * 30
data = data.astype(np.int16)

# Make seeds
seeds = np.zeros([30, 30, 30])
seeds[13:17, 7:10, 5:11] = 1
seeds[0:5:, 0:10, 0:11] = 2

# Run
igc = pspc.ImageGraphCut(data, voxelsize=[1, 1, 1])
igc.set_seeds(seeds)
igc.run()

# Show results
colormap = plt.cm.get_cmap('brg')
colormap._init()
colormap._lut[:1:, 3] = 0

plt.imshow(data[:, :, 10], cmap='gray')
plt.contour(igc.segmentation[:, :, 10], levels=[0.5])
plt.imshow(igc.seeds[:, :, 10], cmap=colormap, interpolation='none')
plt.savefig("gc_example.png")
plt.show()
