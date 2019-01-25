#! /usr/bin/python
# -*- coding: utf-8 -*-


# import funkcí z jiného adresáře
import sys
import os.path
import copy

path_to_script = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(path_to_script, "../extern/pyseg_base/src/"))
# sys.path.append(os.path.join(path_to_script, "../extern/sed3/"))
# sys.path.append(os.path.join(path_to_script, "../src/"))
import unittest


from PyQt4.QtGui import QApplication

import numpy as np


from imcut import pycut

# import imcut.dcmreaddata as dcmr


class PycutTest(unittest.TestCase):
    interactivetTest = False
    # interactivetTest = True

    def generate_data(self, shp=[16, 16, 16], object_type="box"):
        """ Generating random data with cubic object inside"""

        x = np.ones(shp)
        # inserting box
        if object_type == "box":
            x[4:-4, 6:-2, 1:-6] = -1
        elif object_type == "empty_box":
            x[4:-4, 6:-2, 1:-6] = -1
            x[5:-5, 7:-3, 2:-7] = 1
        elif object_type == "wall":
            x[5, :, :] = -3

        x_noisy = x + np.random.normal(0, 0.6, size=x.shape)
        return x_noisy

    @unittest.skipIf(not interactivetTest, "interactiveTest")
    def test_segmentation_with_boundary_penalties(self):
        data_shp = [16, 16, 16]
        # data = self.generate_data(data_shp, boundary_only=True)
        data = self.generate_data(data_shp, object_type="wall")
        seeds = np.zeros(data_shp)
        # setting background seeds
        seeds[:, 0, 0] = 1
        seeds[6, 8:-5, 2] = 2
        # x[4:-4, 6:-2, 1:-6] = -1

        segparams = {"pairwiseAlpha": 10, "use_boundary_penalties": True}
        igc = pycut.ImageGraphCut(data, segparams=segparams)
        igc.interactivity()
        # instead of interacitivity just set seeeds
        # igc.set_seeds(seeds)
        # igc.make_gc()

        # instead of showing just test results
        #        from PyQt4.QtGui import QApplication
        #        app = QApplication(sys.argv)
        #        pyed = seed_editor_qt.QTSeedEditor(igc.segmentation,
        #                            modeFun=self.interactivity_loop,
        #                            voxelVolume=self.voxel_volume,
        #                            seeds=self.seeds, minVal=min_val, maxVal=max_val)
        #        app.exec_()
        # igc.show_segmentation()
        import pdb

        pdb.set_trace()
        segmentation = igc.segmentation
        # Testin some pixels for result
        self.assertTrue(segmentation[0, 0, -1] == 0)
        self.assertTrue(segmentation[7, 9, 3] == 1)
        self.assertTrue(np.sum(segmentation) > 10)

    def test_boundary_penalty_array(self):
        """
        Test if on edge are smaller values
        """

        data = self.generate_data([16, 16, 16]) * 100
        igc = pycut.ImageGraphCut(data)
        # igc.interactivity()

        penalty_array = igc._boundary_penalties_array(axis=0)
        edge_area_pattern = np.mean(penalty_array[3:5, 8:10, 2])
        flat_area_pattern = np.mean(penalty_array[1:3, 3:6, -4:-2])
        self.assertGreater(flat_area_pattern, edge_area_pattern)

    @unittest.skipIf(not interactivetTest, "interactiveTest")
    def test_boundary_penalty(self):
        data = self.generate_data([16, 16, 16]) * 100
        # instead of showing just test results
        # app = QApplication(sys.argv)
        # pyed = seed_editor_qt.QTSeedEditor(data)
        # app.exec_()

        import scipy.ndimage.filters

        # filtered = scipy.ndimage.filters.prewitt(data,0)
        filtered = scipy.ndimage.filters.sobel(data, 0)
        # filtered = scipy.ndimage.filters.gaussian_filter1d(data,sigma=0.6,axis=0, order=1)

        # Oproti Boykov2001b tady nedělím dvojkou. Ta je tam jen proto,
        # aby to slušně vycházelo
        filtered2 = -np.power(filtered, 2) / (512 * np.var(data))
        # Přičítám tu 1024 což je empiricky zjištěná hodnota - aby to dobře vyšlo
        filtered2 = filtered2 + 0  # - np.min(filtered2) + 1e-30
        print("max ", np.max(filtered2))
        print("min ", np.min(filtered2))
        import pdb

        pdb.set_trace()
        # np.exp(-np.random.normal(0

        from seededitorqt import seed_editor_qt
        from PyQt4.QtGui import QApplication

        app = QApplication(sys.argv)
        pyed = seed_editor_qt.QTSeedEditor(filtered2)
        app.exec_()

        filtered3 = np.exp(filtered2)

        pyed = seed_editor_qt.QTSeedEditor(filtered3)
        app.exec_()

        import matplotlib.pyplot as plt

        plt.imshow(filtered3[:, :, 5])
        plt.colorbar()
        plt.show()

    @unittest.skipIf(not interactivetTest, "interactiveTest")
    def test_segmentation(self):
        data_shp = [16, 16, 16]
        data = self.generate_data(data_shp)
        seeds = np.zeros(data_shp)
        # setting background seeds
        seeds[:, 0, 0] = 1
        seeds[6, 8:-5, 2] = 2
        # x[4:-4, 6:-2, 1:-6] = -1

        igc = pycut.ImageGraphCut(data)
        igc.interactivity()
        # instead of interacitivity just set seeeds
        # igc.set_seeds(seeds)
        # igc.make_gc()

        # instead of showing just test results
        #        from PyQt4.QtGui import QApplication
        #        app = QApplication(sys.argv)
        #        pyed = seed_editor_qt.QTSeedEditor(igc.segmentation,
        #                            modeFun=self.interactivity_loop,
        #                            voxelVolume=self.voxel_volume,
        #                            seeds=self.seeds, minVal=min_val, maxVal=max_val)
        #        app.exec_()
        # igc.show_segmentation()
        segmentation = igc.segmentation
        # Testin some pixels for result
        self.assertTrue(segmentation[0, 0, -1] == 0)
        self.assertTrue(segmentation[7, 9, 3] == 1)
        self.assertTrue(np.sum(segmentation) > 10)


#    def setUp(self):
#        #self.dcmdir = os.path.join(path_to_script, '../sample_data/jatra_06mm_jenjatraplus/')
#        self.dcmdir = os.path.join(path_to_script, '../sample_data/jatra_5mm')
#        #self.data3d, self.metadata = dcmr.dcm_read_from_dir(self.dcmdir)
#        reader = dcmr.DicomReader(self.dcmdir)
#        self.data3d = reader.get_3Ddata()
#        self.metadata = reader.get_metaData()

#    def test_DicomReader_overlay(self):
#        #import matplotlib.pyplot as plt
#
#        dcmdir = os.path.join(path_to_script, '../sample_data/volumetrie/')
#        #dcmdir = '/home/mjirik/data/medical/data_orig/jatra-kma/jatra_5mm/'
#        #self.data3d, self.metadata = dcmr.dcm_read_from_dir(self.dcmdir)
#        reader = dcmr.DicomReader(dcmdir)
#        overlay = reader.get_overlay()
#        #import pdb; pdb.set_trace()
#        #plt.imshow(overlay[1][:,:,0])
#        #plt.show()
#
#        self. assertEqual(overlay[1][200,200],1)
#        self. assertEqual(overlay[1][100,100],0)
#    def test_read_volumetry_overlay_with_dicom_module(self):
#        """
#        pydicom module is used for load dicom data. Dicom overlay
#        is saved on (60xx,3000) bit after bit. Data are decoded and
#        each bit is stored as array element.
#        """
#        import dicom
#        # import sed3
#        #import matplotlib.pyplot as plt
#        dcmfile = os.path.join(path_to_script, '../sample_data/volumetrie/volumetry_slice.DCM')
#        data = dicom.read_file(dcmfile)
#
#
#
#        # overlay index
#        i_overlay = 1
#        n_bits = 8
#
#
#        # On (60xx,3000) are stored ovelays.
#        # First is (6000,3000), second (6002,3000), third (6004,3000),
#        # and so on.
#        dicom_tag1 = 0x6000 + 2*i_overlay
#
#        overlay_raw = data[dicom_tag1 ,0x3000].value
#
#        # On (60xx,0010) and (60xx,0011) is stored overlay size
#        rows = data[dicom_tag1,0x0010].value # rows = 512
#        cols = data[dicom_tag1,0x0011].value # cols = 512
#
#        decoded_linear = np.zeros(len(overlay_raw)*n_bits)
#
#        # Decoding data. Each bit is stored as array element
#        for i in range(1,len(overlay_raw)):
#            for k in range (0,n_bits):
#                byte_as_int = ord(overlay_raw[i])
#                decoded_linear[i*n_bits + k] = (byte_as_int >> k) & 0b1
#
#        #overlay = np.array(pol)
#
#        overlay = np.reshape(decoded_linear,[rows,cols])
#
#        #plt.imshow(overlay)
#        #plt.show()
#
#        self. assertEqual(overlay[200,200],1)
#        self. assertEqual(overlay[100,100],0)
#        #pyed = sed3.sed3(overlay)
#        #pyed.show()
#        #import pdb; pdb.set_trace()
#
#
#
#
#
#
#
#    def test_dcmread(self):
#
#        dcmdir = os.path.join(path_to_script, '../sample_data/jatra_5mm')
#        #dcmdir = '/home/mjirik/data/medical/data_orig/jatra-kma/jatra_5mm/'
#        #self.data3d, self.metadata = dcmr.dcm_read_from_dir(self.dcmdir)
#        reader = dcmr.DicomReader(dcmdir)
#        data3d = reader.get_3Ddata()
#        metadata = reader.get_metaData()
##slice size is 512x512
#        self.assertEqual(data3d.shape[0],512)
## voxelsize depth = 5 mm
#        self.assertEqual(metadata['voxelsize_mm'][2],5)
#
#    def test_dcmread_series_number(self):
#
#        dcmdir = os.path.join(path_to_script, '../sample_data/jatra_5mm')
#        #dcmdir = '/home/mjirik/data/medical/data_orig/jatra-kma/jatra_5mm/'
#        #self.data3d, self.metadata = dcmr.dcm_read_from_dir(self.dcmdir)
## spravne cislo serie je 7
#        reader =  dcmr.DicomReader(dcmdir,series_number = 7)
#        data3d = reader.get_3Ddata()
#        metadata = reader.get_metaData()
#        self.assertEqual(data3d.shape[0],512)
#        self.assertEqual(metadata['voxelsize_mm'][2],5)
#
#    @unittest.skipIf(not interactivetTest, 'interactiveTest')
#    def test_dcmread_select_series(self):
#
#        #dirpath = dcmr.get_dcmdir_qt()
#        dirpath = '/home/mjirik/data/medical/data_orig/46328096/'
#        #dirpath = dcmr.get_dcmdir_qt()
#        #app = QMainWindow()
#        reader = dcmr.DicomReader(dirpath, series_number = 55555)#, #qt_app =app)
#        #app.exit()
#        self.data3d = reader.get_3Ddata()
#        self.metadata = reader.get_metaData()
#
#    #@unittest.skipIf(not interactivetTest, 'interactiveTest')
#    @unittest.skip('skip')
#    def test_dcmread_get_dcmdir_qt(self):
#
#        dirpath = dcmr.get_dcmdir_qt()
#        #self.data3d, self.metadata = dcmr.dcm_read_from_dir(self.dcmdir)
#        reader = dcmr.DicomReader(dirpath)
#        self.data3d = reader.get_3Ddata()
#        self.metadata = reader.get_metaData()


if __name__ == "__main__":
    unittest.main()
