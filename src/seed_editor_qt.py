#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SeedEditor for organ segmentation

Example:

$ seed_editor_qp.py -f head.mat
"""

# import unittest
from optparse import OptionParser
from scipy.io import loadmat
import numpy as np
import sys

from PyQt4.QtCore import Qt, QSize, QString
from PyQt4.QtGui import QImage, QDialog,\
    QApplication, QSlider, QPushButton,\
    QLabel, QPixmap, QPainter, qRgba,\
    QComboBox, QIcon, QStatusBar,\
    QHBoxLayout, QVBoxLayout, QFrame, QSizePolicy

# BGRA order
GRAY_COLORTABLE = np.array([[ii, ii, ii, 255] for ii in range(256)],
                           dtype=np.uint8)

SEEDS_COLORTABLE = np.array([[255, 255, 255, 0],
                             [0, 255, 0, 255],
                             [0, 0, 255, 255]], dtype=np.uint8)

CONTOURS_COLORTABLE = np.array([[255, 255, 255, 0],
                                [255, 0, 0, 64]], dtype=np.uint8)

CONTOURLINES_COLORTABLE = np.array([[255, 255, 255, 0],
                                    [255, 0, 0, 16],
                                    [255, 0, 0, 255]], dtype=np.uint8)

VIEW_TABLE = {'axial': (2,1,0),
              'sagittal': (1,0,2),
              'coronal': (2,0,1)}
DRAW_MASK = [
    (np.array([[1]], dtype=np.int8), 'small pen'),
    (np.array([[0, 1, 1, 1, 0],
               [1, 1, 1, 1, 1],
               [1, 1, 1, 1, 1],
               [1, 1, 1, 1, 1],
               [0, 1, 1, 1, 0]], dtype=np.int8), 'middle pen'),
    (np.array([[0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
               [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
               [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
               [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
               [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
               [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
               [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
               [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
               [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
               [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
               [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0]], dtype=np.int8), 'large pen'),
    ]

BOX_BUTTONS_SEED = {
    Qt.LeftButton: 1,
    Qt.RightButton: 2,
    }

BOX_BUTTONS_DRAW = {
    Qt.LeftButton: 1,
    Qt.RightButton: 0,
    }

NEI_TAB = [[-1, -1], [0, -1], [1, -1],
           [-1, 0], [1, 0],
           [-1, 1], [0, 1], [1, 1]]


def erase_reg(arr, p, val=0):
    buff = [p]

    while len(buff) > 0:
        p = buff.pop()
        row = arr[:,p[1],p[2]]
        ii = p[0]
        while ii >= 0:
            if row[ii] <= 0:
                break

            ii -= 1

        ii += 1

        jj = ii
        while jj < arr.shape[0]:
            if row[jj] <= 0:
                break

            row[jj] = val
            jj +=1

        for inb in NEI_TAB:
            irow = p[1] + inb[0]
            islice = p[2] + inb[1]

            if irow >= 0 and irow < arr.shape[1]\
              and islice >= 0 and islice < arr.shape[2]:
                flag = True
                row = arr[:,irow,islice]
                for kk in np.arange(ii, jj):
                    if flag and row[kk] > 0:
                        buff.append((kk, irow, islice))
                        flag = False
                        continue

                    if flag == False and row[kk] <= 0:
                        flag = True

class SliceBox(QLabel):
    """
    Widget for marking reagions of interest in DICOM slices.
    """

    def __init__(self, sliceSize, grid,
                 mode='seeds'):
        """
        Initialize SliceBox.

        Parameters
        ----------
        sliceSize : tuple of int
            Size of slice matrix.
        grid : tuple of float
            Pixel size:
            imageSize = (grid1 * sliceSize1, grid2 * sliceSize2) 
        mode : str
            Editor mode.
        """

        QLabel.__init__(self)

        self.drawing = False
        self.modified = False
        self.seed_mark = None
        self.last_position = None
        self.imagesize = QSize(int(sliceSize[0] * grid[0]),
                               int(sliceSize[1] * grid[1]))
        self.grid = grid
        self.slice_size = sliceSize
        self.ctslice_rgba = None
        self.cw = {'c': 1.0, 'w': 1.0}

        self.seeds = None
        self.contours = None
        self.mask_points = None
        self.erase_region_button = None
        self.erase_fun = None
        self.erase_mode = 'inside'
        self.contour_mode = 'fill'

        if mode == 'draw':
            self.seeds_colortable = CONTOURS_COLORTABLE
            self.box_buttons = BOX_BUTTONS_DRAW
            self.mode_draw = True

        else:
            self.seeds_colortable = SEEDS_COLORTABLE
            self.box_buttons = BOX_BUTTONS_SEED
            self.mode_draw = False

        self.image = QImage(self.imagesize, QImage.Format_RGB32)
        self.setPixmap(QPixmap.fromImage(self.image))
        self.setScaledContents(True)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawImage(event.rect(), self.image)
        painter.end()

    def drawSeedMark(self, x, y):
        xx = self.mask_points[0] + x
        yy = self.mask_points[1] + y
        idx = np.arange(len(xx))
        idx[np.where(xx < 0)] = -1
        idx[np.where(xx >= self.slice_size[0])] = -1
        idx[np.where(yy < 0)] = -1
        idx[np.where(yy >= self.slice_size[1])] = -1
        ii = idx[np.where(idx >= 0)]
        xx = xx[ii]
        yy = yy[ii]

        self.seeds[yy * self.slice_size[0] + xx] = self.seed_mark

    def drawLine(self, p0, p1):
        """
        Draw line to slice image and seed matrix.

        Parameters
        ----------
        p0 : tuple of int
            Line star point.
        p1 : tuple of int
            Line end point.
        """

        x0, y0 = p0
        x1, y1 = p1
        dx = np.abs(x1-x0)
        dy = np.abs(y1-y0)
        if x0 < x1:
            sx = 1
        else:
            sx = -1

        if y0 < y1:
            sy = 1
        else:
            sy = -1

        err = dx - dy

        while True:
            self.drawSeedMark(x0,y0)

            if x0 == x1 and y0 == y1:
                break

            e2 = 2*err
            if e2 > -dy:
                err = err - dy
                x0 = x0 + sx

            if e2 <  dx:
                err = err + dx
                y0 = y0 + sy

    def drawSeeds(self, pos):
        if pos[0] < 0 or pos[0] >= self.slice_size[0] \
                or pos[1] < 0 or pos[1] >= self.slice_size[1]:
            return

        self.drawLine(self.last_position, pos)
        self.updateSlice()

        self.modified = True
        self.last_position = pos

        self.update()

    def get_contours(self, sl):
        sls = sl.reshape(self.slice_size, order='F')
        cnt = sls.copy()
        chunk = np.zeros((cnt.shape[1] + 2,), dtype=np.int8)
        for irow, row in enumerate(sls):
            chunk[1:-1] = row
            chdiff = np.diff(chunk)
            idx1 = np.where(chdiff > 0)[0]
            if idx1.shape[0] > 0:
                idx2 = np.where(chdiff < 0)[0]
                if idx2.shape[0] > 0:
                    cnt[irow,idx1] = 2
                    cnt[irow,idx2 - 1] = 2

        chunk = np.zeros((cnt.shape[0] + 2,), dtype=np.int8)
        for icol, col in enumerate(sls.T):
            chunk[1:-1] = col
            chdiff = np.diff(chunk)
            idx1 = np.where(chdiff > 0)[0]
            if idx1.shape[0] > 0:
                idx2 = np.where(chdiff < 0)[0]
                if idx2.shape[0] > 0:
                    cnt[idx1,icol] = 2
                    cnt[idx2 - 1,icol] = 2

        return cnt.ravel(order='F')

    def composeRgba(self, bg, fg, cmap):
        idxs = fg.nonzero()[0]

        if idxs.shape[0] > 0:
            fg_rgb = cmap[fg[idxs]]

            af = fg_rgb[...,3].astype(np.uint32)
            rgbf = fg_rgb[...,:3].astype(np.uint32)
            rgbb = bg[idxs,:3].astype(np.uint32)

            rgbx = ((rgbf.T * af).T + (rgbb.T * (255 - af)).T) / 255
            bg[idxs,:3] = rgbx.astype(np.uint8)

    def overRgba(self, bg, fg, cmap):
        idxs = fg.nonzero()[0]
        bg[idxs] = cmap[fg[idxs]]

    def updateSlice(self):

        if self.ctslice_rgba is None:
            return

        img = self.ctslice_rgba.copy()

        if self.seeds is not None:
            if self.mode_draw:
                if self.contour_mode == 'fill':
                    self.composeRgba(img, self.seeds,
                                     self.seeds_colortable)
                elif self.contour_mode == 'contours':
                    cnt = self.get_contours(self.seeds)
                    self.composeRgba(img, cnt,
                                     CONTOURLINES_COLORTABLE)

            else:
                self.overRgba(img, self.seeds,
                              self.seeds_colortable)

        if self.contours is not None:
            if self.contour_mode == 'fill':
                self.composeRgba(img, self.contours,
                                 CONTOURS_COLORTABLE)

            elif self.contour_mode == 'contours':
                cnt = self.get_contours(self.contours)
                self.composeRgba(img, cnt,
                                 CONTOURLINES_COLORTABLE)

        image = QImage(img.flatten(),
                     self.slice_size[0], self.slice_size[1],
                     QImage.Format_ARGB32).scaled(self.imagesize)
        painter = QPainter(self.image)
        painter.drawImage(0, 0, image)
        painter.end()

        self.update()

    def getSliceRGBA(self, ctslice):
        if self.cw['w'] > 0:
            mul = 255.0 / float(self.cw['w'])

        else:
            mul = 0

        lb = self.cw['c'] - self.cw['w'] / 2
        aux = (ctslice.ravel(order='F') - lb) * mul
        idxs = np.where(aux < 0)[0]
        aux[idxs] = 0
        idxs = np.where(aux > 255)[0]
        aux[idxs] = 255

        return aux.astype(np.uint8)

    def updateSliceCW(self, ctslice=None):
        if ctslice is not None:
            self.ctslice_rgba = GRAY_COLORTABLE[self.getSliceRGBA(ctslice)]

        self.updateSlice()

    def setSlice(self, ctslice=None, seeds=None, contours=None):
        if ctslice is not None:
            self.ctslice_rgba = GRAY_COLORTABLE[self.getSliceRGBA(ctslice)]

        if seeds is not None:
            self.seeds = seeds.ravel(order='F')
        else:
            self.seeds = None

        if contours is not None:
            self.contours = contours.ravel(order='F')
        else:
            self.contours = None

        self.updateSlice()

    def getSliceSeeds(self):
        if self.modified:
            self.modified = False
            return self.seeds.reshape(self.slice_size, order='F')

        else:
            return None

    def gridPosition(self, pos):
        return (int(pos.x() / self.grid[0]),
                int(pos.y() / self.grid[1]))

    # mouse events
    def mousePressEvent(self, event):
        if event.button() in self.box_buttons:
            self.drawing = True
            self.seed_mark = self.box_buttons[event.button()]
            self.last_position = self.gridPosition(event.pos())

        elif event.button() == Qt.MiddleButton:
            self.drawing = False
            self.erase_region_button = True

    def mouseMoveEvent(self, event):
        if self.drawing:
            self.drawSeeds(self.gridPosition(event.pos()))

    def mouseReleaseEvent(self, event):
        if (event.button() in self.box_buttons) and self.drawing:
            self.drawSeeds(self.gridPosition(event.pos()))
            self.drawing = False

        if event.button() == Qt.MiddleButton\
          and self.erase_region_button == True:
            self.eraseRegion(self.gridPosition(event.pos()),
                             self.erase_mode)

            self.erase_region_button == False

    def resizeSlice(self, new_slice_size=None, new_grid=None):

        if new_slice_size is not None:
            self.slice_size = new_slice_size

        if new_grid is not None:
            self.grid = new_grid

        self.imagesize = QSize(int(self.slice_size[0] * self.grid[0]),
                               int(self.slice_size[1] * self.grid[1]))
        self.image = QImage(self.imagesize, QImage.Format_RGB32)
        self.setPixmap(QPixmap.fromImage(self.image))

    def resizeEvent(self, event):
        new_height = self.height()
        new_grid = new_height / float(self.slice_size[1])
        mul = new_grid / self.grid[1]

        self.grid = np.array(self.grid * mul)
        self.resizeSlice()
        self.updateSlice()

    def leaveEvent(self, event):
        self.drawing = False

    def enterEvent(self, event):
        self.drawing = False

    def setMaskPoints(self, mask):
        self.mask_points = mask

    def getCW(self):
        return self.cw

    def setCW(self, val, key):
        self.cw[key] = val

    def eraseRegion(self, pos, mode):
        if self.erase_fun is not None:
            self.erase_fun(pos, mode)
            self.updateSlice()

    def setEraseFun(self, fun):
        self.erase_fun = fun

class QTSeedEditor(QDialog):
    """
    DICOM viewer.
    """
    @staticmethod
    def get_line(mode='h'):
        line = QFrame()
        if mode == 'h':
            line.setFrameStyle(QFrame.HLine)
        elif mode == 'v':
            line.setFrameStyle(QFrame.VLine)

        line.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Expanding)

        return line

    def initUI(self, shape, vscale, height=600,
               mode='seed'):
        """
        Initialize UI.

        Parameters
        ----------
        shape : (int, int, int)
            Shape of data matrix.
        vscale : (float, float, float)
            Voxel scaling.
        height : int
            Maximal slice height in pixels.
        mode : str
            Editor mode.
        """

        # picture
        grid = height / float(shape[1] * vscale[1])
        mgrid = (grid * vscale[0], grid * vscale[1])
        self.slice_box = SliceBox(shape[:-1], mgrid,
                                  mode)

        # sliders
        self.allow_select_slice = True
        self.n_slices = shape[2]
        self.slider = QSlider(Qt.Vertical)
        self.slider.valueChanged.connect(self.sliderSelectSlice)
        self.slider.label = QLabel()
        self.slider.setRange(1, self.n_slices)

        self.slider_cw = {}
        self.slider_cw['c'] = QSlider(Qt.Horizontal)
        self.slider_cw['c'].valueChanged.connect(self.changeC)
        self.slider_cw['c'].label = QLabel()

        self.slider_cw['w'] = QSlider(Qt.Horizontal)
        self.slider_cw['w'].valueChanged.connect(self.changeW)
        self.slider_cw['w'].label = QLabel()

        self.view_label = QLabel('View size: %d x %d' % self.img_aview.shape[:-1])
        self.voxel_label = QLabel('Voxel size [mm]:\n  %.2f x %.2f x %.2f'\
                                      % tuple(self.voxel_size[np.array(self.act_transposition)]))

        combo_view_options = VIEW_TABLE.keys()
        combo_view = QComboBox(self)
        combo_view.activated[str].connect(self.setView)
        combo_view.addItems(combo_view_options)

        # buttons
        btn_quit = QPushButton("Return", self)
        btn_quit.clicked.connect(self.quit)

        combo_dmask = QComboBox(self)
        combo_dmask.activated.connect(self.changeMask)
        self.mask_points_tab, aux = self.init_draw_mask(DRAW_MASK, mgrid)
        for icon, label in aux:
            combo_dmask.addItem(icon, label)

        self.slice_box.setMaskPoints(self.mask_points_tab[combo_dmask.currentIndex()])

        self.status_bar = QStatusBar()

        vopts = []
        vmenu = []
        appmenu = []
        if mode == 'seed' and self.mode_fun is not None:
            btn_recalc = QPushButton("Recalculate", self)
            btn_recalc.clicked.connect(self.recalculate)
            appmenu.append(QLabel('<b>Segmentation mode</b><br><br><br>' +
                                  'Select the region of interest<br>' +
                                  'using the mouse buttons:<br><br>' +
                                  '&nbsp;&nbsp;<i>left</i> - inner region<br>' +
                                  '&nbsp;&nbsp;<i>right</i> - outer region<br><br>'))
            appmenu.append(btn_recalc)
            appmenu.append(QLabel())
            self.volume_label = QLabel('Volume [mm3]:\n  unknown')
            appmenu.append(self.volume_label)

        if mode == 'seed' or mode == 'crop':
            btn_del = QPushButton("Delete Seeds", self)
            btn_del.clicked.connect(self.deleteSliceSeeds)
            vmenu.append(None)
            vmenu.append(btn_del)

            combo_contour_options = ['fill', 'contours', 'off']
            combo_contour = QComboBox(self)
            combo_contour.activated[str].connect(self.changeContourMode)
            combo_contour.addItems(combo_contour_options)
            self.changeContourMode(combo_contour_options[combo_contour.currentIndex()])
            vopts.append(QLabel('Selection mode:'))
            vopts.append(combo_contour)

        if mode == 'crop':
            btn_crop = QPushButton("Crop", self)
            btn_crop.clicked.connect(self.crop)
            appmenu.append(QLabel('<b>Crop mode</b><br><br><br>' +
                                  'Select the crop region<br>' +
                                  'using the left mouse button<br><br>'))
            appmenu.append(btn_crop)

        if mode == 'draw':
            appmenu.append(QLabel('<b>Manual segmentation<br> mode</b><br><br><br>' +
                                  'Mark the region of interest<br>' +
                                  'using the mouse buttons:<br><br>' +
                                  '&nbsp;&nbsp;<i>left</i> - draw<br>' +
                                  '&nbsp;&nbsp;<i>right</i> - erase<br>' +
                                  '&nbsp;&nbsp;<i>middle</i> - vol. erase<br><br>'))

            btn_reset = QPushButton("Reset", self)
            btn_reset.clicked.connect(self.resetSliceDraw)
            vmenu.append(None)
            vmenu.append(btn_reset)

            combo_erase_options = ['inside', 'outside']
            combo_erase = QComboBox(self)
            combo_erase.activated[str].connect(self.changeEraseMode)
            combo_erase.addItems(combo_erase_options)
            self.changeEraseMode(combo_erase_options[combo_erase.currentIndex()])
            vopts.append(QLabel('Volume erase mode:'))
            vopts.append(combo_erase)

        hbox = QHBoxLayout()
        vbox = QVBoxLayout()
        vbox_left = QVBoxLayout()
        vbox_app = QVBoxLayout()

        hbox.addWidget(self.slice_box)
        hbox.addWidget(self.slider)
        vbox_left.addWidget(self.slider.label)
        vbox_left.addWidget(self.view_label)
        vbox_left.addWidget(self.voxel_label)
        vbox_left.addWidget(QLabel())
        vbox_left.addWidget(QLabel('View plane:'))
        vbox_left.addWidget(combo_view)
        vbox_left.addWidget(self.get_line())
        vbox_left.addWidget(self.slider_cw['c'].label)
        vbox_left.addWidget(self.slider_cw['c'])
        vbox_left.addWidget(self.slider_cw['w'].label)
        vbox_left.addWidget(self.slider_cw['w'])
        vbox_left.addWidget(self.get_line())
        vbox_left.addWidget(QLabel('Drawing mask:'))
        vbox_left.addWidget(combo_dmask)

        for ii in vopts:
            vbox_left.addWidget(ii)

        for ii in vmenu:
            if ii is None:
                vbox_left.addStretch(1)

            else:
                vbox_left.addWidget(ii)

        for ii in appmenu:
            if ii is None:
                vbox_app.addStretch(1)

            else:
                vbox_app.addWidget(ii)

        vbox_app.addStretch(1)
        vbox_app.addWidget(btn_quit)

        hbox.addLayout(vbox_left)
        hbox.addWidget(self.get_line('v'))
        hbox.addLayout(vbox_app)
        vbox.addLayout(hbox)
        vbox.addWidget(self.status_bar)
        self.setLayout(vbox)

        self.setWindowTitle('Segmentation Editor')
        self.show()

    def __init__(self, img, actualSlice=0,
                 seeds=None, contours=None,
                 mode='seed', modeFun=None,
                 voxelSize=[1,1,1]):
        """
        Initiate Editor

        Parameters
        ----------
        img : array
            DICOM data matrix.
        actualSlice : int
            Index of actual slice.
        seeds : array
            Seeds, user defined regions of interest.
        contours : array
            Computed segmentation.
        mode : str
            Editor modes:
               'seed' - seed editor
               'crop' - manual crop
               'draw' - drawing
        modeFun : fun
            Mode function invoked by user button.
        voxelSize : tuple of float
            voxel size [mm]
        """

        QDialog.__init__(self)

        self.mode = mode
        self.mode_fun = modeFun

        self.actual_view = 'axial'
        self.act_transposition = VIEW_TABLE[self.actual_view]
        self.last_view_position = {}
        for ii in VIEW_TABLE.iterkeys():
            self.last_view_position[ii] = img.shape[VIEW_TABLE[ii][-1]] - 1

        self.img = img
        self.img_aview = self.img.transpose(self.act_transposition)
        self.actual_slice = self.img_aview.shape[-1] - actualSlice - 1
        self.last_view_position[self.actual_view] = self.actual_slice

        # set contours
        self.contours = contours
        if self.contours is None:
            self.contours_aview = None

        else:
            self.contours_aview = self.contours.transpose(self.act_transposition)

        self.voxel_size = np.array(voxelSize)
        self.voxel_scale = self.voxel_size / float(np.min(self.voxel_size))
        self.voxel_volume = np.prod(voxelSize)

        # set seeds
        if seeds is None:
            self.seeds = np.zeros(self.img.shape, np.int8)
        else:
            self.seeds = seeds

        self.seeds_aview = self.seeds.transpose(self.act_transposition)
        self.seeds_modified = False

        self.initUI(self.img_aview.shape,
                    self.voxel_scale[np.array(self.act_transposition)],
                    600, mode)

        if mode == 'draw':
            self.seeds_orig = self.seeds.copy()
            self.slice_box.setEraseFun(self.eraseVolume)

        # set view window values C/W
        lb = np.min(img)
        ub = np.max(img)
        dul = ub - lb
        self.cw_range = {'c': [lb, ub], 'w': [1, dul]}
        self.slider_cw['c'].setRange(lb, ub)
        self.slider_cw['w'].setRange(1, dul)
        self.changeC(lb + dul / 2)
        self.changeW(dul)

        self.offsetmm = self.voxel_size * 0

    def showStatus(self, msg):
        self.status_bar.showMessage(QString(msg))
        QApplication.processEvents()

    def init_draw_mask(self, draw_mask, grid):
        mask_points = []
        mask_iconlabel = []
        for mask, label in draw_mask:
            w, h = mask.shape
            xx, yy = mask.nonzero()
            mask_points.append((xx - w/2, yy - h/2))

            img = QImage(w, h, QImage.Format_ARGB32)
            img.fill(qRgba(255, 255, 255, 0))
            for ii in range(xx.shape[0]):
                img.setPixel(xx[ii], yy[ii], qRgba(0, 0, 0, 255))

            img = img.scaled(QSize(w * grid[0], h * grid[1]))
            icon = QIcon(QPixmap.fromImage(img))
            mask_iconlabel.append((icon, label))

        return mask_points, mask_iconlabel

    def saveSliceSeeds(self):
        aux = self.slice_box.getSliceSeeds()
        if aux is not None:
            self.seeds_aview[...,self.actual_slice] = aux
            self.seeds_modified = True

        else:
            self.seeds_modified = False

    def updateCropBounds(self):
        crp = self.getCropBounds()
        if crp is not None:
            _, cri = crp
            self.contours = np.zeros(self.img.shape, np.int8)
            self.contours[cri].fill(1)
            self.contours_aview = self.contours.transpose(self.act_transposition)

    def sliderSelectSlice(self, value):
        self.selectSlice(self.n_slices - value)

    def selectSlice(self, value, force=False):
        if not(self.allow_select_slice):
            return

        if (value < 0) or (value >= self.n_slices):
            return

        if (value != self.actual_slice) or force:
            self.saveSliceSeeds()
            if self.seeds_modified and (self.mode == 'crop'):
                self.updateCropBounds()

        if self.contours is None:
            contours = None

        else:
            contours = self.contours_aview[...,value]

        slider_val = self.n_slices - value
        self.slider.setValue(slider_val)
        self.slider.label.setText('Slice: %d / %d' % (slider_val, self.n_slices))

        self.slice_box.setSlice(self.img_aview[...,value],
                                self.seeds_aview[...,value],
                                contours)
        self.actual_slice = value

    def getSeeds(self):
        return self.seeds

    def getImg(self):
        return self.img

    def getOffset(self):
        return self.offsetmm

    def getSeedsVal(self, label):
        return self.img[self.seeds==label]

    def getContours(self):
        return self.contours

    def setContours(self, contours):
        self.contours = contours
        self.contours_aview = self.contours.transpose(self.act_transposition)

        self.selectSlice(self.actual_slice)

    def changeCW(self, value, key):
        rg = self.cw_range[key]
        if (value < rg[0]) or (value > rg[1]):
            return

        if (value != self.slice_box.getCW()[key]):
            self.slider_cw[key].setValue(value)
            self.slider_cw[key].label.setText('%s: %d' % (key.upper(), value))
            self.slice_box.setCW(value, key)
            self.slice_box.updateSliceCW(self.img_aview[...,self.actual_slice])


    def changeC(self, value):
        self.changeCW(value, 'c')

    def changeW(self, value):
        self.changeCW(value, 'w')

    def setView(self, value):
        self.last_view_position[self.actual_view] = self.actual_slice
        # save seeds
        self.saveSliceSeeds()
        if self.seeds_modified and (self.mode == 'crop'):
            self.updateCropBounds(self.seeds_aview[...,self.actual_slice])

        key = str(value)
        self.actual_view = key
        self.actual_slice = self.last_view_position[key]

        self.act_transposition = VIEW_TABLE[key]
        self.img_aview = self.img.transpose(self.act_transposition)
        self.seeds_aview = self.seeds.transpose(self.act_transposition)

        if self.contours is not None:
            self.contours_aview = self.contours.transpose(self.act_transposition)
            contours = self.contours_aview[...,self.actual_slice]

        else:
            contours = None

        vscale = self.voxel_scale[np.array(self.act_transposition)]
        height = self.slice_box.height()
        grid = height / float(self.img_aview.shape[1] * vscale[1])
        mgrid = (grid * vscale[0], grid * vscale[1])

        self.slice_box.resizeSlice(new_slice_size=self.img_aview.shape[:-1],
                                   new_grid=mgrid)

        self.slice_box.setSlice(self.img_aview[...,self.actual_slice],
                                self.seeds_aview[...,self.actual_slice],
                                contours)

        self.allow_select_slice = False
        self.n_slices = self.img_aview.shape[2]
        slider_val = self.n_slices - self.actual_slice
        self.slider.setValue(slider_val)
        self.slider.setRange(1, self.n_slices)
        self.allow_select_slice = True

        self.slider.label.setText('Slice: %d / %d' % (slider_val,
                                                      self.n_slices))
        self.view_label.setText('View size: %d x %d' % self.img_aview.shape[:-1])

    def changeMask(self, val):
        self.slice_box.setMaskPoints(self.mask_points_tab[val])

    def changeContourMode(self, val):
        self.slice_box.contour_mode = str(val)
        self.slice_box.updateSlice()

    def changeEraseMode(self, val):
        self.slice_box.erase_mode = str(val)

    def eraseVolume(self, pos, mode):
        self.showStatus("Processing...")
        xyz = pos + (self.actual_slice,)
        p = tuple(np.array(xyz)[np.array(self.act_transposition)])
        if self.seeds[p] > 0:
            if mode == 'inside':
                erase_reg(self.seeds, p, val=0)

            elif mode == 'outside':
                erase_reg(self.seeds, p, val=-1)
                idxs = np.where(self.seeds < 0)
                self.seeds.fill(0)
                self.seeds[idxs] = 1

        if self.contours is None:
            contours = None

        else:
            contours = self.contours_aview[...,self.actual_slice]

        self.slice_box.setSlice(self.img_aview[...,self.actual_slice],
                                self.seeds_aview[...,self.actual_slice],
                                contours)

        self.showStatus("Done")

    def cropUpdate(self, img):

        for ii in VIEW_TABLE.iterkeys():
            self.last_view_position[ii] = 0
        self.actual_slice = 0

        self.img = img
        self.img_aview = self.img.transpose(self.act_transposition)

        self.contours = None
        self.contours_aview = None

        self.seeds = np.zeros(self.img.shape, np.int8)
        self.seeds_aview = self.seeds.transpose(self.act_transposition)
        self.seeds_modified = False

        vscale = self.voxel_scale[np.array(self.act_transposition)]
        height = self.slice_box.height()
        grid = height / float(self.img_aview.shape[1] * vscale[1])
        mgrid = (grid * vscale[0], grid * vscale[1])

        self.slice_box.resizeSlice(new_slice_size=self.img_aview.shape[:-1],
                                   new_grid=mgrid)

        self.slice_box.setSlice(self.img_aview[...,self.actual_slice],
                                self.seeds_aview[...,self.actual_slice],
                                None)

        self.allow_select_slice = False
        self.n_slices = self.img_aview.shape[2]
        self.slider.setValue(self.actual_slice + 1)
        self.slider.setRange(1, self.n_slices)
        self.allow_select_slice = True

        self.slider.label.setText('Slice: %d / %d' % (self.actual_slice + 1,
                                                      self.n_slices))
        self.view_label.setText('View size: %d x %d' % self.img_aview.shape[:-1])

    def getCropBounds(self):

        nzs = self.seeds.nonzero()
        cri = []
        flag = True
        for ii in range(3):
            if nzs[ii].shape[0] == 0:
                flag = False
                break

            smin, smax = np.min(nzs[ii]), np.max(nzs[ii])
            if smin == smax:
                flag = False
                break

            cri.append((smin, smax))

        if flag:
            cri = np.array(cri)

            out = []
            offset = []
            for jj, ii in enumerate(cri):
                out.append(slice(ii[0], ii[1] + 1))
                offset.append(ii[0] * self.voxel_size[jj])

            return np.array(offset), tuple(out)

        else:
            return None

    def crop(self):
        self.showStatus("Processing...")

        crp = self.getCropBounds()

        if crp is not None:
            offset, cri = crp
            crop = self.img[cri]
            self.img = np.ascontiguousarray(crop)
            self.offsetmm += offset

            self.showStatus('Done')

        else:
            self.showStatus('Region not selected!')

        self.cropUpdate(self.img)

    def recalculate(self, event):
        self.saveSliceSeeds()
        if np.abs(np.min(self.seeds) - np.max(self.seeds)) < 2:
            self.showStatus('Inner and outer regions not defined!')
            return

        self.showStatus("Processing...")
        self.mode_fun(self)
        self.selectSlice(self.actual_slice)
        self.updateVolume()
        self.showStatus("Done")

    def deleteSliceSeeds(self, event):
        self.seeds_aview[...,self.actual_slice] = 0
        self.slice_box.setSlice(seeds=self.seeds_aview[...,self.actual_slice])
        self.slice_box.updateSlice()

    def resetSliceDraw(self, event):
        seeds_orig_aview = self.seeds_orig.transpose(self.act_transposition)
        self.seeds_aview[...,self.actual_slice] = seeds_orig_aview[...,self.actual_slice]
        self.slice_box.setSlice(seeds=self.seeds_aview[...,self.actual_slice])
        self.slice_box.updateSlice()

    def quit(self, event):
        self.close()

    def updateVolume(self):
        text = 'Volume [mm3]:\n  unknown'
        if self.voxel_volume is not None:
            if self.mode == 'draw':
                vd = self.seeds

            else:
                vd = self.contours

            if vd is not None:
                nzs = vd.nonzero()
                nn = nzs[0].shape[0]
                text = 'Volume [mm3]:\n  %.2e' % (nn * self.voxel_volume)

        self.volume_label.setText(text)

def gen_test():
    test = {}
    test['data'] = np.zeros((10,10,4), dtype=np.uint8)
    test['voxelsize_mm'] = (2, 2, 2.5)

    return test

usage = '%prog [options]\n' + __doc__.rstrip()
help = {
    'in_file': 'input *.mat file with "data" field',
    'mode': '"seed", "crop" or "draw" mode',
    #'out_file': 'store the output matrix to the file',
    #'debug': 'run in debug mode',
    'gen_test': 'generate test data',
    'test': 'run unit test',
}

def main():
    parser = OptionParser(description='Segmentation editor')
    parser.add_option('-f','--filename', action='store',
                      dest='in_filename', default=None,
                      help=help['in_file'])
    # parser.add_option('-d', '--debug', action='store_true',
    #                   dest='debug', help=help['debug'])
    parser.add_option('-m', '--mode', action='store',
                      dest='mode', default='seed', help=help['mode'])
    parser.add_option('-t', '--tests', action='store_true',
                      dest='unit_test', help=help['test'])
    parser.add_option('-g', '--gener_data', action='store_true',
                      dest='gen_test', help=help['gen_test'])

    # parser.add_option('-o', '--outputfile', action='store',
    #                   dest='out_filename', default='output.mat',
    #                   help=help['out_file'])
    (options, args) = parser.parse_args()

    if options.gen_test:
        dataraw = gen_test()

    else:
        if options.in_filename is None:
            raise IOError('No input data!')

        else:
            dataraw = loadmat(options.in_filename,
                              variable_names=['data', 'segdata',
                                              'voxelsize_mm', 'seeds'])
            if not ('segdata' in dataraw):
                dataraw['segdata'] = None

    app = QApplication(sys.argv)
    pyed = QTSeedEditor(dataraw['data'],
                        seeds=dataraw['segdata'],
                        mode=options.mode,
                        voxelSize=dataraw['voxelsize_mm'])

    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
