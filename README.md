PYSEG_BASE

About
-----

Segmentation tools based on the graph cut algorithm. You can 
see video to get an idea.

    http://youtu.be/bFSyY4jyMHw


Links
-----

  https://github.com/mjirik/pyseg_base

Authors
-------

Miroslav Jirik
Vladimir Lukes

License
-------

New BSD License, see the LICENSE file.

Install
-------

    pip install pyseg_base

See INSTALL file for more information

Run
---

Create output.mat file:
    
    python pysegbase/dcmreaddata.py -i directoryWithDicomFiles --degrad 4
    
See data:

    python pysegbase/seed_editor_qt.py -f output.mat
    
Make graph_cut:

    python pysegbase/pycut.py -f output.mat


Use is as a library:

    import numpy as np
    import pyseg_base as psb
    
    data = np.random.rand(30, 30, 30)
    data[10:20, 5:15, 3:13] += 1
    data = data * 30
    data = data.astype(np.int16)
    igc = psb.ImageGraphCut(data, voxelsize=[1, 1, 1])
    igc.interactivity()
