
About
-----

Segmentation tools based on the graph cut algorithm. You can 
see video to get an idea.

[![Graph-Cut segmentation](https://img.youtube.com/vi/bFSyY4jyMHw/0.jpg)](https://www.youtube.com/watch?v=bFSyY4jyMHw)

please cite:

    @INPROCEEDINGS{jirik2013,
        author = {Jirik, M. and Lukes, V. and Svobodova, M. and Zelezny, M.},
        title = {Image Segmentation in Medical Imaging via Graph-Cuts.},
        year = {2013},
        journal = {11th International Conference on Pattern Recognition and Image Analysis: New Information Technologies (PRIA-11-2013). Samara, Conference Proceedings },
        url = {http://www.kky.zcu.cz/en/publications/JirikMmjirik_2013_ImageSegmentationin},
    }




Authors
-------

* Miroslav Jirik
* Vladimir Lukes

Special requirements
-----

See third party licenses

 * [gco_python](https://github.com/amueller/gco_python)
 * [gco-v3.0](http://vision.csd.uwo.ca/code/gco-v3.0.zip) 

Resources
-----

  https://github.com/mjirik/pysegbase
  
  https://github.com/amueller/gco_python


License
-------

New BSD License, see the LICENSE file.

Install conda
----

    conda install -c mjirik -c conda-forge pysegbase
    pip install pygco

Install pip
-------

    pip install pygco pysegbase

See INSTALL file for more information

Run
---

Create output.mat file:
    
    python pysegbase/dcmreaddata.py -i directoryWithDicomFiles --degrad 4
    
See data:

    python pysegbase/seed_editor_qt.py -f output.mat
    
Make graph_cut:

    python pysegbase/pycut.py -i output.mat


Use is as a library:

    import numpy as np
    import pysegbase.pycut as pspc

    data = np.random.rand(30, 30, 30)
    data[10:20, 5:15, 3:13] += 1
    data = data * 30
    data = data.astype(np.int16)
    igc = pspc.ImageGraphCut(data, voxelsize=[1, 1, 1])
    seeds = igc.interactivity()
    
![pysegbase_screenshot](http://147.228.240.61/queetech/www/pysegbase_screenshot0.png)

    
More complex example without interactivity
---

    import numpy as np
    import pysegbase.pycut as pspc
    import matplotlib.pyplot as plt

    # create data
    data = np.random.rand(30, 30, 30)
    data[10:20, 5:15, 3:13] += 1
    data = data * 30
    data = data.astype(np.int16)
    
    # Make seeds
    seeds = np.zeros([30,30,30])
    seeds[13:17, 7:10, 5:11] = 1
    seeds[0:5:, 0:10, 0:11] = 2
    
    # Run 
    igc = pspc.ImageGraphCut(data, voxelsize=[1, 1, 1])
    igc.set_seeds(seeds)
    igc.run()
    
    # Show results
    colormap = plt.cm.get_cmap('brg')
    colormap._init()
    colormap._lut[:1:,3]=0
    
    plt.imshow(data[:, :, 10], cmap='gray') 
    plt.contour(igc.segmentation[:, :,10], levels=[0.5])
    plt.imshow(igc.seeds[:, :, 10], cmap=colormap, interpolation='none')
    plt.show()


![example_img](https://raw.githubusercontent.com/mjirik/pyseg_base/master/imgs/example_result.png)


Configuration
===


        segparams = {
            # 'method':'graphcut',
            'method': 'graphcut',
            'use_boundary_penalties': False,
            'boundary_dilatation_distance': 2,
            'boundary_penalties_weight': 1,
            'modelparams': {
                'type': 'gmmsame',
                'fv_type': "fv_extern",
                'fv_extern': fv_function,
                'adaptation': 'original_data',
            }
            'mdl_stored_file': False,
        }
        
*mdl_stored_file*: if this is set, load model from file 

[read more about configuration](https://github.com/mjirik/pysegbase/blob/master/pysegbase/pycut.py)
