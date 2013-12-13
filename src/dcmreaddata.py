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

import traceback

import logging
logger = logging.getLogger(__name__)


__version__ = [1,2]

def obj_from_file(filename='annotation.yaml', filetype='yaml'):
    """
    Read object from file
    """
    if filetype == 'yaml':
        import yaml
        f = open(filename, 'r')
        obj = yaml.load(f)
    elif filetype == 'pickle':
        import pickle
        f = open(filename, 'rb')
        obj = pickle.load(f)
    else:
        logger.error('Unknown filetype')

    f.close()

    return obj

def obj_to_file(obj, filename='annotation.yaml', filetype='yaml'):
    """
    Writes annotation in file
    """
    if filetype == 'yaml':
        import yaml
        f = open(filename, 'w')
        yaml.dump(obj, f)
    elif filetype == 'pickle':
        import pickle
        f = open(filename, 'wb')
        pickle.dump(obj, f, -1)
    else:
        logger.error('Unknown filetype')

    f.close

class DicomReader():
    """
    Example:

    dcr = DicomReader(os.path.abspath(dcmdir))
    data3d = dcr.get_3Ddata()
    metadata = dcr.get_metaData()

    """
    dicomdir_filename = 'dicomdir.pkl'
    def __init__(self, dirpath=None, initdir='.',
                 qt_app=None, gui=True, series_number = None):
        self.valid = False
        self.dirpath = dirpath
        self.dcmdir = self.get_dir()
        self.series_number = series_number
        self.overlay = {}
        self.dcmlist = []

        if len(self.dcmdir) > 0:
            self.valid = True
            counts, bins = self.status_dir()

            if len (bins) > 1:
                if self.series_number == None:
                    if (qt_app is not None) or gui:
                        from PyQt4.QtGui import QInputDialog
                        sbins = ', '.join([str(ii) for ii in bins])
                        snstring, ok = \
                            QInputDialog.getText(qt_app,
                                                 'Select serie:',
                                                 'Select serie [%s]:' % sbins,
                                                 text='%d' % bins[0])
                    else:
                        series_info = self.dcmdirstats()
                        print self.print_series_info(series_info)
                        snstring = raw_input ('Select Serie: ')

                    sn = int(snstring)
                else:
                    sn = self.series_number

            else:
                sn = bins[0]

            self.series_number = sn

            self.dcmlist = self.get_sortedlist(SeriesNumber=sn)

    def validData(self):
        return self.valid

    def get_overlay(self):
        """
        Function make 3D data from dicom file slices. There are usualy
        more overlays in the data.
        """
        overlay = {}
        dcmlist = self.dcmlist

        for i  in range(len(dcmlist)):
            onefile = dcmlist[i]
            logger.info(onefile)
            data = dicom.read_file(onefile)

            if len(overlay) == 0:
# first there is created dictionary with avalible overlay indexes
                for i_overlay in range (0,50):
                    try:
                        # overlay index
                        data2d = self.decode_overlay_slice(data,i_overlay)
                        #import pdb; pdb.set_trace()
                        shp2 = data2d.shape
                        overlay[i_overlay]= np.zeros([len(dcmlist), shp2[0],
                                                      shp2[1] ], dtype=np.int8)
                        overlay[i_overlay][-i-1,:,:] = data2d

                    except:
                        #print "nefunguje", i_overlay
                        pass

            else:
                for i_overlay in overlay.keys():
                        data2d = self.decode_overlay_slice(data,i_overlay)
                        overlay[i_overlay][-i-1,:,:] = data2d

        return overlay

    def decode_overlay_slice(self, data, i_overlay):
            # overlay index
            n_bits = 8


            # On (60xx,3000) are stored ovelays.
            # First is (6000,3000), second (6002,3000), third (6004,3000),
            # and so on.
            dicom_tag1 = 0x6000 + 2*i_overlay

            overlay_raw = data[dicom_tag1 ,0x3000].value

            # On (60xx,0010) and (60xx,0011) is stored overlay size
            rows = data[dicom_tag1,0x0010].value # rows = 512
            cols = data[dicom_tag1,0x0011].value # cols = 512

            decoded_linear = np.zeros(len(overlay_raw)*n_bits)

            # Decoding data. Each bit is stored as array element
# TODO neni tady ta jednička blbě?
            for i in range(1,len(overlay_raw)):
                for k in range (0,n_bits):
                    byte_as_int = ord(overlay_raw[i])
                    decoded_linear[i*n_bits + k] = (byte_as_int >> k) & 0b1

            #overlay = np.array(pol)
            overlay_slice = np.reshape(decoded_linear,[rows,cols])
            return overlay_slice

    def get_3Ddata(self):
        """
        Function make 3D data from dicom file slices
        """
        data3d = []
        dcmlist = self.dcmlist

        for i in range(len(dcmlist)):
            onefile = dcmlist[i]
            logger.info(onefile)
            data = dicom.read_file(onefile)
            data2d = data.pixel_array
            #import pdb; pdb.set_trace()

            if len(data3d) == 0:
                shp2 = data2d.shape
                data3d = np.zeros([len(dcmlist), shp2[0], shp2[1]],
                                  dtype=np.int16)

            try:
                new_data2d = (np.float(data.RescaleSlope) * data2d)\
                    + np.float(data.RescaleIntercept)

            except:
                logger.warning('problem with RescaleSlope and RescaleIntercept')
            # first readed slide is at the end

            data3d[-i-1,:,:] = new_data2d

            logger.debug("Data size: " + str(data3d.nbytes)\
                    + ', shape: ' + str(shp2) +'x'+ str(len(dcmlist)) )

        return data3d

    def get_metaData(self, dcmlist=None, ifile=0):
        """
        Get metadata.
        Voxel size is obtained from PixelSpacing and difference of
        SliceLocation of two neighboorhoding slices (first have index ifile).
        Files in are used.
        """
        if dcmlist == None:
            dcmlist = self.dcmlist

        if len(dcmlist) <= 0:
            return {}

        data = dicom.read_file(dcmlist[ifile])
        try:
            data2 = dicom.read_file(dcmlist[ifile+1])
            voxeldepth = float(np.abs(data.SliceLocation - data2.SliceLocation ))
        except:
            logger.warning('Problem with voxel depth. Using SliceThickness,'\
                               + ' SeriesNumber: ' + str(data.SeriesNumber))

            try:
                voxeldepth = float(data.SliceThickness)
            except:
                logger.warning('Probem with SliceThicknes, setting zero. '\
                                   + traceback.format_exc())
                voxeldepth = 0

        pixelsize_mm = data.PixelSpacing
        voxelsize_mm = [
                voxeldepth,
                float(pixelsize_mm[0]),
                float(pixelsize_mm[1]),
                ]
        metadata = {'voxelsize_mm': voxelsize_mm,
                'Modality': data.Modality,
                'SeriesNumber':self.series_number
                }

        try:
            metadata['SeriesDescription'] = data.SeriesDescription

        except:
            logger.warning(
                'Problem with tag SeriesDescription, SeriesNumber: ' +
                str(data.SeriesNumber))
        try:
            metadata['ImageComments'] = data.ImageComments
        except:
            logger.warning(
                'Problem with tag ImageComments, SeriesNumber: ' +
                str(data.SeriesNumber))
        try:
            metadata['Modality'] = data.Modality
        except:
            logger.warning(
                'Problem with tag Modality, SeriesNumber: ' +
                str(data.SeriesNumber))

        metadata['dcmfilelist'] = self.dcmlist

        #import pdb; pdb.set_trace()
        return metadata

    def dcmdirstats(self):
        """ Dicom series staticstics, input is dcmdir, not dirpath
        Information is generated from dicomdir.pkl and first files of series
        """
        import numpy as np
        dcmdir = self.dcmdir
        # get series number
# vytvoření slovníku, kde je klíčem číslo série a hodnotou jsou všechny
# informace z dicomdir
        series_info = {line['SeriesNumber']:line for line in dcmdir}

# počítání velikosti série
        try:
            dcmdirseries = [line['SeriesNumber'] for line in dcmdir]

        except:
            logger.debug('Dicom tag SeriesNumber not found')
            series_info = {0:{'Count':0}}
            return series_info
            #return [0],[0]

        bins = np.unique(dcmdirseries)
        binslist = bins.tolist()
#  kvůli správným intervalům mezi biny je nutno jeden přidat na konce
        mxb = np.max(bins)+1
        binslist.append(mxb)
        #binslist.insert(0,-1)
        counts, binsvyhodit = np.histogram(dcmdirseries, bins = binslist)

        #pdb.set_trace();

        # sestavení informace o velikosti série a slovníku

        for i in range(0,len(bins)):
            series_info[bins[i]]['Count']=counts[i]

            # adding information from files
            lst = self.get_sortedlist(SeriesNumber = bins[i])
            metadata = self.get_metaData(dcmlist = lst)
# adding dictionary metadata to series_info dictionary
            series_info[bins[i]] = dict(
                    series_info[bins[i]].items() +
                    metadata.items()
                    )

        return series_info

    def print_series_info(self, series_info):
        """
        Print series_info from dcmdirstats
        """
        strinfo = ''
        if len (series_info) > 1:
            for serie_number in series_info.keys():
                strl = str(serie_number) + " ("\
                    + str(series_info[serie_number]['Count'])
                try:
                    strl = strl + ", "\
                        + str( series_info[serie_number]['Modality'])
                    strl = strl + ", "\
                        + str( series_info[serie_number]['SeriesDescription'])
                    strl = strl + ", "\
                        + str( series_info[serie_number]['ImageComments'])
                except:
                    logger.debug('Tag Modlity or ImageComment not found in dcminfo')
                    pass

                strl = strl + ')'
                strinfo = strinfo + strl + '\n'
                #print strl

        return strinfo

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
        createdcmdir = True

        dicomdirfile = os.path.join(self.dirpath,self.dicomdir_filename)
        ftype='pickle'
        # if exist dicomdir file and is in correct version, use it
        if os.path.exists(dicomdirfile):
            dcmdirplus = obj_from_file(dicomdirfile, ftype)
            try:
                if dcmdirplus ['version'] == __version__:
                    createdcmdir = False
                dcmdir = dcmdirplus['filesinfo']
            except:
                logger.debug('Found dicomdir.pkl with wrong version')
                pass


        if createdcmdir:
            dcmdirplus = self.create_dir()
            dcmdir = dcmdirplus['filesinfo']
            if (writedicomdirfile) and len(dcmdir) > 0:
                obj_to_file(dcmdirplus, dicomdirfile, ftype)
                #obj_to_file(dcmdir, dcmdiryamlpath )

        dcmdir = dcmdirplus['filesinfo']
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
                metadataline = {'filename' : teil,
                              'InstanceNumber' : dcmdata.InstanceNumber,
                              'SeriesNumber' : dcmdata.SeriesNumber,
                              'AcquisitionNumber' : dcmdata.AcquisitionNumber
                              }

                #try:
                #    metadataline ['ImageComment'] = dcmdata.ImageComments
                #    metadataline ['Modality'] = dcmdata.Modality
                #except:
                #    print 'Problem with ImageComments and Modality tags'

                files.append(metadataline)

            except Exception as e:
                if head !=  self.dicomdir_filename:
                    print 'Dicom read problem with file ' + filepath

        files.sort(key=lambda x: x['InstanceNumber'])
        files.sort(key=lambda x: x['SeriesNumber'])
        files.sort(key=lambda x: x['AcquisitionNumber'])

        dcmdirplus = {'version':__version__, 'filesinfo':files}
        return dcmdirplus

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
        dcmdir = QFileDialog.getExistingDirectory(
                caption='Select DICOM Folder',
                options=QFileDialog.ShowDirsOnly)
    else:
        app = QApplication(sys.argv)
        dcmdir = QFileDialog.getExistingDirectory(
                caption='Select DICOM Folder',
                options=QFileDialog.ShowDirsOnly)
        #app.exec_()
        app.exit(0)
    if len(dcmdir) > 0:

        dcmdir = "%s" %(dcmdir)
        dcmdir = dcmdir.encode("utf8")
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
                                   'voxelsize_mm': metadata['voxelsize_mm']})

    print "Data size: %d, shape: %s" % (data3d.nbytes, data3d.shape)
