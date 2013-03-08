#! /usr/bin/python
# -*- coding: utf-8 -*-
"""
DICOM reader

Example:

$ dcmreaddata -d sample_data -o head.mat
"""

import sys
import os
import dicom
import numpy as np
from optparse import OptionParser
from scipy.io import savemat

import logging
logger = logging.getLogger(__name__)

def obj_from_file(filename='annotation.yaml', filetype='yaml'):
    """
    Read object from file
    """
    f = open(filename, 'r')
    if filetype == 'yaml':
        import yaml
        obj = yaml.load(f)
    elif filetype == 'pickle':
        import pickle
        obj = pickle.load(f)
    else:
        logger.error('Unknown filetype')
        
    f.close()

    return obj

def obj_to_file(obj, filename='annotation.yaml', filetype='yaml'):
    """
    Writes annotation in file
    """
    f = open(filename, 'w')
    if filetype == 'yaml':
        import yaml
        yaml.dump(obj, f)
    elif filetype == 'pickle':
        import pickle
        pickle.dump(obj, f, -1)
    else:
        logger.error('Unknown filetype')

    f.close

class DicomReader():
    def __init__(self, dirpath=None, initdir='.'):    
        self.valid = False
        self.dirpath = dirpath
        self.dcmdir = self.get_dir()

        if len(self.dcmdir) > 0:
            self.valid = True
            counts, bins = self.status_dir()

            if len (bins) > 1:
                snstring = raw_input ('Select Serie: ')
                sn = int(snstring)
            else:
                sn = bins[0]

            self.dcmlist = self.get_sortedlist(SeriesNumber=sn)

    def validData(self):
        return self.valid
        
    def get_metaData(self, ifile=0):
        """
        Get metadata.
        Voxel size is obtained from PixelSpacing and difference of 
        SliceLocation of two neighboorhoding slices (first have index ifile).
        """

        data = dicom.read_file(self.dcmlist[ifile])
        try:
            data2 = dicom.read_file(self.dcmlist[ifile+1])
            voxeldepth = float(np.abs(data.SliceLocation - data2.SliceLocation ))
        except:
            logger.warning('Problem with voxel depth. Using SliceThickness')
            voxeldepth = float(data.SliceThickness)

        
        pixelsizemm = data.PixelSpacing
        voxelsizemm = [float(pixelsizemm[0]),
                       float(pixelsizemm[1]),
                       voxeldepth]
        metadata = {'voxelsizemm': voxelsizemm, 'Modality': data.Modality}

        #import pdb; pdb.set_trace()
        return metadata

    def get_3Ddata(self):
        """
        Function make 3D data from dicom file slices
        """
        data3d = []
        dcmlist = self.dcmlist

        for i  in range(len(dcmlist)):
            onefile = dcmlist[i]
            logger.info(onefile)
            data = dicom.read_file(onefile)
            data2d = data.pixel_array

            if len(data3d) == 0:
                shp2 = data2d.shape
                data3d = np.zeros([shp2[0], shp2[1], len(dcmlist)],
                                  dtype=np.int16)
            else:
                data3d [:,:,i] = data2d

                logger.debug("Data size: " + str(data3d.nbytes)\
                                 + ', shape: ' + str(shp2) +'x'+ str(len(dcmlist)) )

        return data3d

    def files_in_dir(self, dirpath, wildcard="*", startpath=None):
        """
        Function generates list of files from specific dir
        
        files_in_dir(dirpath, wildcard="*.*", startpath=None)

        dirpath: required directory
        wilcard: mask for files
        startpath: start for relative path

        Example
        files_in_dir('medical/jatra-kiv','*.dcm', '~/data/')
        """

        import glob

        filelist = []

        if startpath != None:
            completedirpath = os.path.join(startpath, dirpath)
        else:
            completedirpath = dirpath

        if os.path.exists(completedirpath):
            logger.info('completedirpath = '  + completedirpath)

        else:
            logger.error('Wrong path: '  + completedirpath)
            raise Exception('Wrong path : ' + completedirpath)

        for infile in glob.glob(os.path.join(completedirpath, wildcard)):
            filelist.append(infile)

        if len(filelist) == 0:
            logger.error('No required files in path: '  + completedirpath)
            raise Exception ('No required file in path: ' + completedirpath)

        return filelist

    def get_dir(self, writedicomdirfile=True):
        """
        Check if exists dicomdir file and load it or cerate it

        dcmdir = get_dir(dirpath)

        dcmdir: list with filenames, SeriesNumber, InstanceNumber and 
        AcquisitionNumber
        """

        dicomdirfile = os.path.join(self.dirpath, 'dicomdir.pkl')
        ftype='pickle'

        if os.path.exists(dicomdirfile):
            dcmdir = obj_from_file(dicomdirfile, ftype)

        else:
            dcmdir = self.create_dir()
            if (writedicomdirfile) and len(dcmdir) > 0:
                obj_to_file(dcmdir, dicomdirfile, ftype)

        return dcmdir

    def create_dir(self):
        """
        Function crates list of all files in dicom dir with all IDs
        """

        filelist = self.files_in_dir(self.dirpath)
        files=[]

        for filepath in filelist:
            head, teil = os.path.split(filepath)
            try:
                dcmdata=dicom.read_file(filepath)
                files.append({'filename' : teil, 
                              'InstanceNumber' : dcmdata.InstanceNumber,
                              'SeriesNumber' : dcmdata.SeriesNumber,
                              'AcquisitionNumber' : dcmdata.AcquisitionNumber
                              })

            except Exception as e:
                print 'Dicom read problem with file ' + filepath

        files.sort(key=lambda x: x['InstanceNumber'])
        files.sort(key=lambda x: x['SeriesNumber'])
        files.sort(key=lambda x: x['AcquisitionNumber'])

        return files

    def status_dir(self):
        """input is dcmdir, not dirpath """

        try:
            dcmdirseries = [line['SeriesNumber'] for line in self.dcmdir]
        except:
            return [0],[0]

        bins = np.unique(dcmdirseries)
        binslist = bins.tolist()
        #  kvůli správným intervalům mezi biny je nutno jeden přidat na konce
        mxb = np.max(bins)+1
        binslist.append(mxb)
        counts, binsvyhodit = np.histogram(dcmdirseries, bins = binslist)

        return counts, bins

    def get_sortedlist(self, startpath="", SeriesNumber=None):
        """
        Function returns sorted list of dicom files. File paths are organized by
        SeriesUID, StudyUID and FrameUID

        Example:
        get_sortedlist()
        get_sortedlist('~/data/')
        """
        dcmdir = self.dcmdir[:]
        dcmdir.sort(key=lambda x: x['InstanceNumber'])
        dcmdir.sort(key=lambda x: x['SeriesNumber'])
        dcmdir.sort(key=lambda x: x['AcquisitionNumber'])

        # select sublist with SeriesNumber
        if SeriesNumber != None:
            dcmdir = [line for line in dcmdir if line['SeriesNumber']==SeriesNumber]

        logger.debug('SeriesNumber: ' +str(SeriesNumber))

        filelist = []
        for onefile in dcmdir:
            filelist.append(os.path.join(startpath, self.dirpath, onefile['filename']))
            head, tail = os.path.split(onefile['filename'])

        return filelist

def get_dcmdir_qt(app=False):
    from PyQt4.QtGui import QFileDialog, QApplication

    if app:
        dcmdir = QFileDialog.getExistingDirectory(caption='Select DICOM Folder',
                                                  options=QFileDialog.ShowDirsOnly)
        
    else:
        app = QApplication(sys.argv)
        dcmdir = QFileDialog.getExistingDirectory(caption='Select DICOM Folder',
                                                  options=QFileDialog.ShowDirsOnly)
        app.exit()

    if len(dcmdir) > 0:
        dcmdir = str(dcmdir)

    else:
        dcmdir = None    

    return dcmdir

usage = '%prog [options]\n' + __doc__.rstrip()
help = {
    'dcm_dir': 'DICOM data direcotory',
    'out_file': 'store the output matrix to the file',
}

if __name__ == "__main__":
    parser = OptionParser(description='Read DIOCOM data.')
    parser.add_option('-d','--dcmdir', action='store',
                      dest='dcmdir', default=None,
                      help=help['dcm_dir'])
    parser.add_option('-o', '--outputfile', action='store',
                      dest='out_filename', default='output.mat',
                      help=help['out_file'])
    (options, args) = parser.parse_args()

    logger.setLevel(logging.WARNING)
    ch = logging.StreamHandler()
    logger.addHandler(ch)

    if options.dcmdir == None:
        dcmdir = get_dcmdir_qt()
        if dcmdir is None:
            raise IOError('No DICOM directory!')
    else:
        dcmdir = options.dcmdir

    dcr = DicomReader(os.path.abspath(dcmdir))
    data3d = dcr.get_3Ddata()
    metadata = dcr.get_metaData()
    savemat(options.out_filename, {'data': data3d,
                                   'voxelsizemm': metadata['voxelsizemm']})

    print "Data size: %d, shape: %s" % (data3d.nbytes, data3d.shape)
