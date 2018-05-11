#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Arne F. Meyer <arne.f.meyer@gmail.com>
# License: GPLv3

"""
    base class for object trackers and widgets
"""


from __future__ import print_function

import abc
import os
import os.path as op
import collections
from functools import partial
import itertools
import time
import copy
import traceback

import numpy as np
import matplotlib.pyplot as plt
import pyqtgraph as pg
import tqdm

from ..util.opencv import cv2, cv
from ..util.pyqt import QtCore, QtGui
from ..util.pyqt import qwidgets as qw
from ..util.system import makedirs_save
from ..io.video import dump_video_to_memmap_file, get_first_frame
from ..util.roi import select_ROI_and_mask


# -----------------------------------------------------------------------------
# helper functions
# -----------------------------------------------------------------------------

GRAY_COLOR_TABLE = [QtGui.qRgb(i, i, i) for i in range(256)]


def toQImage(im, copy=False):
    """https://gist.github.com/smex/5287589"""

    if im is None:
        return QtGui.QImage()

    if im.dtype == np.uint8:

        if len(im.shape) == 2:
            qim = QtGui.QImage(im.data, im.shape[1], im.shape[0],
                               im.strides[0], QtGui.QImage.Format_Indexed8)
            qim.setColorTable(GRAY_COLOR_TABLE)
            return qim.copy() if copy else qim

        elif len(im.shape) == 3:

            if im.shape[2] == 3:
                qim = QtGui.QImage(im.data, im.shape[1], im.shape[0],
                                   im.strides[0], QtGui.QImage.Format_RGB888)
                return qim.copy() if copy else qim

            elif im.shape[2] == 4:
                qim = QtGui.QImage(im.data, im.shape[1], im.shape[0],
                                   im.strides[0], QtGui.QImage.Format_ARGB32)
                return qim.copy() if copy else qim

    else:
        return None


class ClickablePlotWidget(pg.PlotWidget):

    def __init__(self, *args, **kwargs):

        super(ClickablePlotWidget, self).__init__(*args, **kwargs)

    def plot(self, x, y):

        self.plotItem.clear()
        self.plotDataItem = self.plotItem.plot(x, y, clickable=True)
        scene = self.plotDataItem.scene()
        scene.sigMouseClicked.connect(self.clicked)

    def clicked(self, event):

        mod = event.modifiers()

        if mod == QtCore.Qt.ControlModifier:

            items = self.plotDataItem.scene().items(event.scenePos())
            views = [x for x in items if isinstance(x, pg.ViewBox)]

            if len(views) > 0:
                view = views[0]
                pt = self.plotDataItem.mapFromItem(view, event.pos())
                x, y = self.plotDataItem.getData()
                print("point:", pt)

            event.accept()


# -----------------------------------------------------------------------------
# tracker base classes
# -----------------------------------------------------------------------------


class InvalidNumberOfObjectsException(Exception):
    pass


class TrackerParameter():

    def __init__(self, name, value, range=(1, 100),
                 description=None):

        self.name = name
        self.value = value
        self.range = range
        self.description = description

    def copy_from(self, obj):

        # TODO: replace by simple copy of the whole object?
        self.name = copy.copy(obj.name)
        self.value = copy.copy(obj.value)
        self.range = copy.copy(obj.range)
        self.description = copy.copy(obj.description)

    def copy_to(self, obj):

        obj.name = copy.copy(self.name)
        obj.value = copy.copy(self.value)
        obj.range = copy.copy(self.range)
        obj.description = copy.copy(self.description)


class ParameterAutomation():

    def __init__(self, param):

        self.parameter = param
        self.default_value = param.value

        self.timestamps = []
        self.values = []

    def add_value(self, ts, value):

        if ts in self.timestamps:
            self.values[self.timestamps.index(ts)] = value
        else:
            self.timestamps.append(ts)
            self.values.append(value)

    def get_value(self, ts, return_default=True):

        ind = np.where(np.asarray(self.timestamps) <= ts)[0]

        if len(ind) > 0:
            return self.values[ind[-1]]
        else:
            if return_default:
                return self.default_value
            else:
                return self.parameter.value

    def get_values(self, ts):

        values = np.zeros_like(ts)
        timestamps = np.asarray(self.timestamps)
        for i in range(1, len(timestamps)):
            v = np.logical_and(ts >= timestamps[i-1],
                               ts < timestamps[i])
            values[v] = self.values[i]

        return values


class TrackedObject(object):

    def __init__(self, pos, **kwargs):

        self.pos = pos
        self.annotations = {}

        if len(kwargs) > 0:
            self.annotations.update(**kwargs)

    @abc.abstractmethod
    def draw(self, frame):
        return


class TrackedEllipse(TrackedObject):

    def __init__(self, pos, axes, angle=0, color=[255, 0, 0], **kwargs):

        super(TrackedEllipse, self).__init__(pos, **kwargs)

        self.axes = axes
        self.angle = angle
        self.color = color

    def draw(self, frame, draw_center=True, color=None):

        if color is None:
            color = self.color

        cv2.ellipse(frame,
                    tuple(np.array(self.pos, dtype=np.int)),
                    tuple(np.array(self.axes, dtype=np.int)),
                    self.angle, 0, 360, color, thickness=1)

        if draw_center:
            cv2.circle(frame, tuple(np.array(self.pos, dtype=np.int)),
                       1, color, 0)


class AbstractTracker(object):

    __meta__ = abc.ABCMeta

    def __init__(self, file_path,
                 bbox=None,
                 mask=None,
                 max_num_frames=-1,
                 overwrite=False,
                 widget=None,
                 oversample=None,
                 timestamps=None,
                 max_num_objects=np.Inf):

        if not op.exists(file_path):
            raise OSError("File does not exists:", file_path)

        if not op.isfile(file_path):
            raise OSError("Given path is not a file:", file_path)

        if not op.splitext(file_path)[1] == '.mp4':
            raise TypeError("Currently only mp4 video files are supported")

        self.file_path = file_path
        self.max_num_frames = max_num_frames
        self.mask = mask
        self.widget = widget
        self.oversample = oversample
        self.max_num_objects = max_num_objects

        self.mmap = dump_video_to_memmap_file(file_path,
                                              output=None,
                                              bbox=bbox,
                                              max_num_frames=max_num_frames,
                                              timestamps=timestamps)
        self.bbox = self.mmap['bbox']

        self._open_memmap_file()

        self._user_mask = None
        self._boundary_contour = None

        self.parameters = self.init_parameters()

    @abc.abstractmethod
    def init_parameters(self):
        """must return a list of TrackerParameter objects"""
        return

    @abc.abstractmethod
    def get_main_clip(self):
        """the name of the main frame/clip"""
        return

    @abc.abstractmethod
    def get_extra_clips(self):
        """a list with names of optional extra frames/clips"""
        return

    @abc.abstractmethod
    def save_objects(filepath, objects):
        """write a list of objects to a file"""
        return

    def set_parameter(self, name, value):

        p = self.get_parameter(name)

        if value >= p.range[0] and value <= p.range[1]:
            p.value = value
        else:
            raise ValueError('Value outside valid range')

    def get_parameter(self, name):

        return [p for p in self.parameters if p.name == name][0]

    def save_parameters(self, file_path, automation=None,
                        widget_parameters=None):

        params = dict([(p.name, p.value) for p in self.parameters])

        auto = {}
        if automation is not None:
            for a in automation:
                auto[a.parameter.name] = {'timestamps': a.timestamps,
                                          'values': a.values}

        np.savez(file_path,
                 parameters=params,
                 automation=auto,
                 widget_parameters=widget_parameters)

    def load_parameters(self, file_path):

        data = np.load(file_path)

        for name in data:
            self.get_parameter(name).value = data[name]

    def cleanup(self):

        self._close_memmap_file()

        if op.exists(self.mmap['file']):
            os.remove(self.mmap['file'])

    def _open_memmap_file(self):

        self.mmap['fp'] = np.memmap(self.mmap['file'],
                                    dtype=self.mmap['dtype'],
                                    mode='r',
                                    shape=self.mmap['size'])

    def _close_memmap_file(self):

        if 'fp' in self.mmap:
            del self.mmap['fp']

    def load_frame(self, index=None):

        if index is None:
            index = self.index

        if 'fp' in self.mmap:
            frame = np.copy(self.mmap['fp'][index, :, :])
        else:
            frame = None

        return frame

    @property
    def timestamps(self):

        return self.mmap['timestamps']

    @property
    def n_frames(self):

        return self.mmap['n_frames']

    @property
    def boundary_contour(self):

        return self._boundary_contour

    @boundary_contour.setter
    def boundary_contour(self, contour):

        if contour is not None and 'height' in self.mmap:

            # only update user mask when it changed
            if self._boundary_contour is None or \
                    len(contour) != len(self._boundary_contour) or \
                    not np.all(contour == self._boundary_contour) or \
                    self._user_mask is None:

                w, h = self.mmap['width'], self.mmap['height']

                w = w*self.oversample
                h = h*self.oversample
                self._user_mask = np.zeros((h, w), dtype=np.uint8)
                for i in range(w):
                    for j in range(h):
                        val = cv2.pointPolygonTest(contour, (i, j), False)
                        if val > 0:
                            self._user_mask[j, i] = 1
        else:
            self._user_mask = None

        self._boundary_contour = contour

    def set_boundary_contour(self, contour):

        self.boundary_contour = contour

# -----------------------------------------------------------------------------
# GUI-related classes
# -----------------------------------------------------------------------------


class FrameLabel(QtGui.QLabel):

    bbox_selected = QtCore.pyqtSignal(tuple)
    path_selected = QtCore.pyqtSignal(np.ndarray)

    def __init__(self):

        super(FrameLabel, self).__init__()

        self.setFrameStyle(QtGui.QFrame.StyledPanel)
        self.image = None

        self.pixmap = None
        self.scaled_pixmap = None

        self.origin = QtCore.QPoint()
        self.rubberband = QtGui.QRubberBand(QtGui.QRubberBand.Rectangle, self)

        self.select_rectangle = True
        self.points = []

    def set_image(self, img):

        self.image = img
        self.repaint()

    def paintEvent(self, event):

        if self.image is not None:

            self.pixmap = QtGui.QPixmap.fromImage(toQImage(self.image))

            try:
                size = self.size()
                painter = QtGui.QPainter(self)
                point = QtCore.QPoint(0, 0)
                trans_mode = QtCore.Qt.SmoothTransformation
                pm = self.pixmap.scaled(size, QtCore.Qt.KeepAspectRatio,
                                        transformMode=trans_mode)

                painter.drawPixmap(point, pm)
                self.scaled_pixmap = pm

                if len(self.points) > 0:
                    painter.setPen(QtCore.Qt.red)
                    for p in self.points:
                        painter.drawPoint(p.x(), p.y())

            except BaseException:
                # TODO: figure out why this causes problems on mac osx and
                # not on linux-based systems
                pass

    def mousePressEvent(self, event):

        if event.button() == QtCore.Qt.LeftButton:

            if self.select_rectangle:
                self.origin = QtCore.QPoint(event.pos())
                self.rubberband.setGeometry(QtCore.QRect(self.origin,
                                                         QtCore.QSize()))
                self.rubberband.show()
            else:
                self.points = []

    def mouseMoveEvent(self, event):

            if self.select_rectangle:
                if not self.origin.isNull():
                    pos = event.pos()
                    self.rubberband.setGeometry(QtCore.QRect(self.origin,
                                                             pos).normalized())
            else:
                self.points.append(event.pos())
                self.repaint()

    def mouseReleaseEvent(self, event):

        if event.button() == QtCore.Qt.LeftButton:

            if self.pixmap is None or self.scaled_pixmap is None:
                print("pixmap or scaled pixmap == None. "
                      "Probably some PyQt repaint-related error.")
                return

            if self.select_rectangle:

                self.rubberband.hide()

                rect = self.rubberband.geometry()
                _w = self.scaled_pixmap.width()
                _h = self.scaled_pixmap.height()

                w = self.pixmap.width()
                h = self.pixmap.height()

                x = int(rect.left() / float(_w) * w)
                y = int(rect.top() / float(_h) * h)
                ww = int(rect.width() / float(_w) * w)
                hh = int(rect.height() / float(_h) * h)

                bbox = (x, y, ww, hh)

                self.bbox_selected.emit(bbox)

            else:

                _w = self.scaled_pixmap.width()
                _h = self.scaled_pixmap.height()

                w = self.pixmap.width()
                h = self.pixmap.height()

                points = np.zeros((len(self.points), 2))
                for i, p in enumerate(self.points):
                    points[i, 0] = p.x() / float(_w) * w
                    points[i, 1] = p.y() / float(_h) * h

                self.path_selected.emit(points)


class PlaybackHandler(QtCore.QObject):

    # playback control states
    NOTHING = 0
    RUNNING = 1
    PAUSING = 2
    EXITING = 3

    finished = QtCore.pyqtSignal()
    updated = QtCore.pyqtSignal(int)

    def __init__(self, n_frames, frame_rate=10.):

        super(PlaybackHandler, self).__init__()

        self.n_frames = n_frames
        self.frame_rate = frame_rate
        self.status = self.NOTHING
        self.current_frame = 0
        self.mutex = QtCore.QMutex()

    @QtCore.pyqtSlot()
    def process(self):

        n_frames = self.n_frames

        while True:

            self.mutex.lock()
            status = self.status
            frame_rate = self.frame_rate
            self.mutex.unlock()

            if status == self.RUNNING:

                if self.current_frame < n_frames:

                    self.updated.emit(self.current_frame)
                    self.current_frame += 1

                time.sleep(1./frame_rate)

            else:

                if status == self.EXITING:
                    break
                else:
                    time.sleep(0.5)

        self.current_frame = 0

        self.finished.emit()

    def set_status(self, s):

        self.mutex.lock()
        self.status = s
        self.mutex.unlock()

    def get_status(self):

        self.mutex.lock()
        s = self.status
        self.mutex.unlock()

        return s

    def set_current_frame(self, x):

        self.mutex.lock()
        self.current_frame = x
        self.mutex.unlock()

    def set_frame_rate(self, x):

        self.mutex.lock()
        self.frame_rate = x
        self.mutex.unlock()


class AbstractTrackerWidget(qw.QWidget):
    """Simple Qt-based widget for object tracking"""

    __meta__ = abc.ABCMeta

    # assigned modes
    MODE_PLAYING = 0
    MODE_ASSIGNING = 1
    MODE_EDITING = 2
    MODE_REMOVING = 3
    MODE_ADDING = 4
    MODE_SELECTING = 5

    AUTOMATION_NONE = 1
    AUTOMATION_REC = 2
    AUTOMATION_PLAY = 3

    def __init__(self, file_path,
                 max_num_frames=-1,
                 param_file=None,
                 overwrite=False,
                 bbox=None,
                 output=None,
                 oversample=None,
                 frame_rate=10,
                 suffix='_tracked_data'):

        qw.QWidget.__init__(self)

        self.file_path = file_path
        self.max_num_frames = max_num_frames
        self.output = output
        self.suffix = suffix

        self.current_index = 0
        self.frame_rate = frame_rate
        self.mode = self.MODE_PLAYING
        self.slider_is_moving = False

        self.thread = None

        # the pupil tracker object is doing the actual work
        if bbox is None:
            frame = get_first_frame(file_path)
            bbox, mask = select_ROI_and_mask(frame)

        self.tracker = self.create_tracker(file_path, bbox=bbox,
                                           oversample=oversample)

        if param_file is not None and op.exists(param_file):
            self.tracker.load_parameters(param_file)

        # quantities to plot
        n_frames = self.tracker.n_frames
        self.objects = n_frames * [[]]

        # to keep track of different parameter settings
        self.automation_mode = AbstractTrackerWidget.AUTOMATION_NONE

        self.tracker_parameters = []
        for i, p in enumerate(self.tracker.parameters):
            self.tracker_parameters.append(ParameterAutomation(p))

        self.widget_parameters = {'boundary_contour': {'index': [],
                                                       'timestamps': [],
                                                       'contours': []}}

        # GUI stuff
        self.setWindowTitle('Pupil Tracker')

        # have a couple of columns
        layout = qw.QHBoxLayout(self)

        # box for main stuff
        vbox = qw.QVBoxLayout(self)

        # control buttons
        vbox.addWidget(qw.QLabel("<b>Controls</b>", self))
        self.buttons = collections.OrderedDict()
        self.buttons['start'] = qw.QPushButton('Play', self)
        self.buttons['start'].clicked.connect(self.start)

        self.buttons['stop'] = qw.QPushButton('Stop', self)
        self.buttons['stop'].clicked.connect(self.stop)

        self.buttons['reset'] = qw.QPushButton('Reset position', self)
        self.buttons['reset'].clicked.connect(self.reset)

        self.buttons['save'] = qw.QPushButton('Save', self)
        self.buttons['save'].clicked.connect(self.save_results)

        self.buttons['exit'] = qw.QPushButton('Exit', self)
        self.buttons['exit'].clicked.connect(self.close_and_exit)

        hbox = qw.QHBoxLayout(self)
        [hbox.addWidget(self.buttons[k]) for k in self.buttons]
        vbox.addLayout(hbox)

        vbox.addWidget(self.HLine())

        # fps slider
        vbox.addWidget(qw.QLabel("<b>Frame rate</b>", self))
        self.fps_slider = qw.QSlider(QtCore.Qt.Horizontal)
        self.fps_slider.setMinimum(1)
        self.fps_slider.setMaximum(60)
        self.fps_slider.setValue(self.frame_rate)
        self.fps_slider.setTickPosition(qw.QSlider.TicksBelow)
        self.fps_slider.setTickInterval(10)
        self.fps_slider.sliderMoved.connect(self.fps_slider_changed)
        vbox.addWidget(self.fps_slider)

        vbox.addWidget(self.HLine())

        # position slider
        vbox.addWidget(qw.QLabel("<b>Position</b>", self))
        n_frames = self.tracker.n_frames
        self.pos_slider = qw.QSlider(QtCore.Qt.Horizontal)
        self.pos_slider.setMinimum(0)
        self.pos_slider.setMaximum(n_frames-1)
        self.pos_slider.setValue(0)
        self.pos_slider.setTickPosition(qw.QSlider.TicksBelow)
        self.pos_slider.setTickInterval(int(n_frames / 25.))
        self.pos_slider.valueChanged.connect(self.pos_slider_changed)
        self.pos_slider.sliderMoved.connect(self.pos_slider_moved)
        self.pos_slider.sliderPressed.connect(self.pos_slider_pressed)
        self.pos_slider.sliderReleased.connect(self.pos_slider_released)
        self.pos_slider.setTracking(True)
        vbox.addWidget(self.pos_slider)

        vbox.addWidget(self.HLine())

        # ellipse fitting parameters
        vbox.addWidget(qw.QLabel("<b>Fitting parameters</b>", self))
        grid = QtGui.QGridLayout()

        def update_fitting_parameter(name, label, value):
            self.tracker.set_parameter(name, value)
            label.setText(str(value))

            self.update_gui(self.current_index, update_slider=False)

        self.sliders = {}
        for i, param in enumerate(self.tracker.parameters):

            slider = qw.QSlider(QtCore.Qt.Horizontal)

            slider.setMinimum(param.range[0])
            slider.setMaximum(param.range[1])
            slider.setValue(param.value)

            num = QtGui.QLabel(str(param.value))

            slider.valueChanged.connect(partial(update_fitting_parameter,
                                                param.name, num))

            label = QtGui.QLabel(param.name)
            if param.description is not None:
                label.setToolTip(param.description)

            grid.addWidget(label, i, 0)
            grid.addWidget(num, i, 1)
            grid.addWidget(slider, i, 2)

            self.sliders[param.name] = slider

        vbox.addLayout(grid)

        layout.addLayout(vbox)

        # processing options
        vbox.addWidget(qw.QLabel("<b>Processing</b>", self))
        hbox = qw.QHBoxLayout(self)

        button_proc = qw.QPushButton('Process frames', self)
        button_proc.setToolTip("Process all frames using current parameters")
        button_proc.clicked.connect(self.process_all_frames)
        hbox.addWidget(button_proc)

        button_rem = QtGui.QPushButton('Remove mode', self)
        button_rem.setToolTip("Manually remove objects")
        button_rem.setCheckable(True)
        button_add = QtGui.QPushButton('Add mode', self)
        button_add.setToolTip("Manually add objects (not implemented)")
        button_add.setCheckable(True)
        button_bound = QtGui.QPushButton('Boundary mode', self)
        button_bound.setToolTip("Set boundary for detecting objects")
        button_bound.setCheckable(True)

        button_rem.clicked.connect(partial(self.edit_mode_changed,
                                           'remove'))
        button_add.clicked.connect(partial(self.edit_mode_changed,
                                           'add'))
        button_bound.clicked.connect(partial(self.edit_mode_changed,
                                             'bound'))
        hbox.addWidget(button_rem)
        hbox.addWidget(button_add)
        hbox.addWidget(button_bound)

        button = QtGui.QPushButton('Jump to next', self)
        button.setToolTip("Jump to next frame with mulitple objects in "
                          "remove mode")
        button.clicked.connect(self.jump_to_next_editing_position)
        hbox.addWidget(button)

        self.buttons = {'remove': button_rem,
                        'add': button_add,
                        'bound': button_bound,
                        'process': button_proc}

        vbox.addLayout(hbox)

#        # automation options (not fully working)
#        vbox.addWidget(qw.QLabel("<b>Parameter automation</b>", self))
#        hbox = qw.QHBoxLayout(self)
#
#        button_rec = qw.QPushButton('Record', self)
#        button_rec.setCheckable(True)
#        button_rec.clicked.connect(partial(self.automation_changed, 'rec',
#                                           button_rec))
#        hbox.addWidget(button_rec)
#
#        button_play = qw.QPushButton('Play', self)
#        button_play.setCheckable(True)
#        button_play.clicked.connect(partial(self.automation_changed, 'play',
#                                            button_play))
#        hbox.addWidget(button_play)
#
#        button_reset = qw.QPushButton('Reset', self)
#        button_reset.clicked.connect(partial(self.automation_changed, 'reset',
#                                             button_reset))
#        hbox.addWidget(button_reset)
#
#        combo = qw.QComboBox(self)
#        for p in self.tracker.parameters:
#            combo.addItem(p.name)
#        combo.currentIndexChanged.connect(self.comboChanged)
#        hbox.addWidget(combo)
#
#        vbox.addLayout(hbox)
#
#        self.buttons['params_rec'] = button_rec
#        self.buttons['params_play'] = button_play
#        self.buttons['params_reset'] = button_reset

        # parameter plot
        self.current_param_index = 0
        self.param_widget = ClickablePlotWidget()
        ts = self.tracker.timestamps
        p0 = self.tracker.parameters[self.current_param_index]
        self.param_widget.plot(ts, p0.value*np.ones_like(ts))
        vbox.addWidget(self.param_widget)

        # plotting options
        vbox.addWidget(qw.QLabel("<b>Plotting</b>", self))
        hbox = qw.QHBoxLayout(self)

        button = qw.QPushButton('Plot', self)
        button.clicked.connect(self.plot_results)
        hbox.addWidget(button)

        button = qw.QPushButton('Hist', self)
        button.clicked.connect(self.show_hist)
        hbox.addWidget(button)

        vbox.addLayout(hbox)

        # number of objects
        self.plot_widget = pg.PlotWidget()
        item = self.plot_widget.getPlotItem()
        ts = self.tracker.timestamps
        item.plot(ts, np.zeros_like(ts))
        self.plot_item = item
        vbox.addWidget(self.plot_widget)

        vbox.addStretch()

        # video frames
        grid = QtGui.QGridLayout()

        self.clips = {}

        ef = FrameLabel()
        ef.setFixedWidth(600)
        ef.bbox_selected.connect(partial(self.bbox_selected, 'main'))
        ef.path_selected.connect(partial(self.path_selected, 'main'))
        grid.addWidget(ef, 0, 0, 1, 3)

        self.clips[self.tracker.get_main_clip()] = ef

        for i, k in enumerate(self.tracker.get_extra_clips()):

            fl = FrameLabel()
            fl.setFixedWidth(200)
            grid.addWidget(fl, 1, i, 1, 1)
            self.clips[k] = fl

        layout.addLayout(grid)

        # widget settings
        self.setLayout(layout)
        self.setGeometry(100, 100, 500, 1000)

        self.start_thread()

    def __del__(self):

        if self.thread is not None:
            self.handler.set_status(PlaybackHandler.EXITING)
            self.thread.wait()

        self.tracker.cleanup()

    @abc.abstractmethod
    def create_tracker(self, file_path):
        pass

    def HLine(self):

        hline = qw.QFrame()
        hline.setFrameShape(qw.QFrame.HLine)
        hline.setFrameShadow(qw.QFrame.Sunken)

        return hline

    def resizeEvent(self, event):

        pass

    def fps_slider_changed(self):

        fps = self.fps_slider.value()
        self.frame_rate = fps
        if self.handler is not None:
            self.handler.set_frame_rate(fps)

    def pos_slider_changed(self):

        pos = self.pos_slider.value()
        self.update_gui(pos, update_slider=False)

        if self.handler is not None:
            self.handler.set_current_frame(pos)

    def pos_slider_moved(self):

        pos = self.pos_slider.value()
        self.update_gui(pos, update_slider=False)

        if self.handler is not None:
            self.handler.set_current_frame(pos)

    def pos_slider_pressed(self):

        if self.handler is not None:
            self.handler.set_status(PlaybackHandler.NOTHING)
            self.slider_is_moving = True

    def pos_slider_released(self):

        self.slider_is_moving = False

    def shortcut_pressed(self, name):

        pass

    def edit_mode_changed(self, which, enabled):

        buttons = self.buttons

        if enabled:

            if which == 'add':
                self.mode = self.MODE_ADDING
                [buttons[b].setChecked(False) for b in ['remove', 'bound']]

                frame = self.clips[self.tracker.get_main_clip()]
                frame.select_rectangle = False

            elif which == 'remove':
                self.mode = self.MODE_REMOVING
                [buttons[b].setChecked(False) for b in ['add', 'bound']]

                frame = self.clips[self.tracker.get_main_clip()]
                frame.select_rectangle = True

            elif which == 'bound':
                self.mode = self.MODE_SELECTING
                [buttons[b].setChecked(False) for b in ['add', 'remove']]
                frame = self.clips[self.tracker.get_main_clip()]
                frame.select_rectangle = False

        else:
            self.mode = self.MODE_ASSIGNING

    def automation_changed(self, which, button):

        buttons = self.buttons

        if which == 'rec':

            if button.isChecked():
                self.automation_mode = self.AUTOMATION_REC
                buttons['params_rec'].setChecked(True)
                buttons['params_play'].setChecked(False)
            else:
                self.automation_mode = self.AUTOMATION_NONE
                buttons['params_rec'].setChecked(False)

        elif which == 'play':

            if button.isChecked():
                self.automation_mode = self.AUTOMATION_PLAY
                buttons['params_rec'].setChecked(False)
                buttons['params_play'].setChecked(True)
            else:
                self.automation_mode = self.AUTOMATION_NONE
                buttons['params_play'].setChecked(False)

        elif which == 'reset':

            self.automation_mode = self.AUTOMATION_NONE
            buttons['params_rec'].setChecked(False)
            buttons['params_play'].setChecked(False)

            self.tracker_parameters = []
            for i, p in enumerate(self.tracker.parameters):
                self.tracker_parameters.append(ParameterAutomation(p))

            cont = self.widget_parameters['boundary_contour']
            cont['index'] = []
            cont['timestamps'] = []

    def comboChanged(self, index):

        self.current_param_index = index
        item = self.param_widget
        p = self.tracker_parameters[index]
        item.plot(self.tracker.timestamps,
                  p.get_values(self.tracker.timestamps))

    def bbox_selected(self, name, bbox):
        """rempove ellipse(s) with center inside bbox"""

        index = self.current_index
        good_objects = []
        for obj in self.objects[index]:

            x0, y0 = obj.pos
            if x0 >= bbox[0] and x0 <= bbox[0] + bbox[2] and \
                    y0 >= bbox[1] and y0 <= bbox[1] + bbox[3]:
                pass

            else:
                good_objects.append(obj)

        self.objects[index] = good_objects

        self.update_gui(index, update_slider=False)

    def path_selected(self, name, points):
        """fit ellipse to selected path"""

        if self.mode == self.MODE_ADDING:

            print("Adding mode not implemented")

        elif self.mode == self.MODE_SELECTING:

            contour = np.zeros((points.shape[0]+1, 1, points.shape[1]),
                               dtype=np.int32)
            contour[:-1, 0, :] = points.astype(np.int32)
            contour[-1, 0, :] = contour[0, 0, :]
            self.tracker.boundary_contour = contour

            self.buttons['bound'].setChecked(False)
            self.mode = self.MODE_PLAYING

    def start_thread(self):

        self.thread = QtCore.QThread(objectName='PlaybackThread')
        self.handler = PlaybackHandler(self.tracker.n_frames,
                                       frame_rate=self.frame_rate)
        self.handler.set_status(PlaybackHandler.NOTHING)
        self.handler.moveToThread(self.thread)
        self.handler.finished.connect(self.thread.quit)
        self.handler.updated.connect(self.update_gui)
        self.thread.started.connect(self.handler.process)
        self.thread.start()

    def update_gui(self, index, update_slider=True):

        self.current_index = index
        ts = self.tracker.timestamps[index]

        if self.automation_mode == self.AUTOMATION_REC:

            for i, p in enumerate(self.tracker.parameters):
                self.tracker_parameters[i].add_value(ts, p.value)

            cont = self.widget_parameters['boundary_contour']

            if self.tracker.boundary_contour not in \
                    cont['contours']:

                # add new contour for current frame
                cont['contours'].append(
                    self.tracker.boundary_contour)

                found = [1 if np.all(self.tracker.boundary_contour == C) else 0
                         for C in cont['contours']]
                cont['index'].append(
                    np.where(found)[0])
                cont['timestamps'].append(ts)

        elif self.automation_mode == self.AUTOMATION_PLAY:

            params = [p.get_value(ts) for p in self.tracker_parameters]
            sliders = self.sliders

            for i, par in enumerate(self.tracker.parameters):

                if params[i] != par.value:
                    par.value = params[i]
                    slider = sliders[par.name]
                    slider.setValue(par.value)

            cont = self.widget_parameters['boundary_contour']
            if len(cont) > 0:

                cont_ind = cont['index']
                contours = cont['contours']
                cont_ts = np.asarray(cont['timestamps'])
                ind = np.where(cont_ts <= ts)[0]
                if len(ind) > 0:
                    # find last valid contour
                    self.tracker.boundary_contour = contours[cont_ind[ind[-1]]]
                else:
                    self.tracker.boundary_contour = None

        if self.mode < self.MODE_EDITING:

            objects, clips = self.tracker.process_frame(index=index,
                                                        draw=True)

            k = self.tracker.get_main_clip()
            frame_obj = self.draw_objects(index,
                                          frame=clips[k],
                                          objects=objects)
            self.clips[k].set_image(frame_obj)

            for k in self.tracker.get_extra_clips():
                self.clips[k].set_image(clips[k])

            self.objects[index] = objects
            self.current_index = index

        else:
            _, clips = self.tracker.process_frame(index=index,
                                                  draw=True)
            objects = self.objects[index]

            k = self.tracker.get_main_clip()
            frame_obj = self.draw_objects(index,
                                          frame=clips[k],
                                          objects=objects)
#            frame_obj = clips[k]
            self.clips[k].set_image(frame_obj)

            for k in self.tracker.get_extra_clips():

                self.clips[k].set_image(clips[k])

        if update_slider:
            self.pos_slider.setValue(index)

        cnt = np.asarray([len(obj) for obj in self.objects])
        self.plot_item.clear()
        self.plot_item.plot(self.tracker.timestamps, cnt)

        item = self.param_widget
        p = self.tracker_parameters[self.current_param_index]
        item.plot(self.tracker.timestamps,
                  p.get_values(self.tracker.timestamps))

    def draw_objects(self, index, frame=None, objects=None, color=None,
                     draw_center=True):

        if index is None:
            index = self.current_index

        if frame is None:
            # read frame and convert to gray scale
            frame_gray = self.tracker.load_frame(index)
            if frame_gray is None:
                return

            frame = cv2.cvtColor(frame_gray, cv.CV_GRAY2RGB)

        frame_objects = np.copy(frame)

        cycler = itertools.cycle([[255, 0, 0],
                                  [0, 255, 0],
                                  [0, 0, 255]])

        if objects is None:
            objects = self.objects[index]

        for obj in objects:

            if color is None:
                c = next(cycler)
            else:
                c = color

            obj.color = c
            obj.draw(frame_objects)

        return frame_objects

    def jump_to_next_editing_position(self):

        if self.mode >= self.MODE_EDITING:

            cnt = np.asarray([len(obj) for obj in self.objects])

            if self.mode == self.MODE_REMOVING:
                ind = np.where(cnt > 1)[0]

            elif self.mode == self.MODE_ADDING:
                ind = np.where(cnt < 1)[0]

            if len(ind) > 0:
                ind_after = ind[ind > self.current_index]
                if len(ind_after) > 0:
                    self.update_gui(ind_after[0])
                else:
                    self.update_gui(ind[0])

    def start(self):

        if self.handler is not None:
            self.handler.set_status(PlaybackHandler.RUNNING)

    def stop(self):

        if self.handler is not None:
            self.handler.set_status(PlaybackHandler.NOTHING)

    def reset(self):

        if self.handler is not None:

            self.handler.set_status(PlaybackHandler.NOTHING)
            self.handler.set_current_frame(0)
            self.update_gui(0, update_slider=True)

    def show_hist(self):

        cnt = np.asarray([len(obj) for obj in self.objects])
        N = cnt.sum()

        if N > 0:

            pos = np.zeros((N, 2))
            ii = 0
            for objs in self.objects:
                for obj in objs:
                    pos[ii, :] = obj.pos
                    ii += 1

            fig, ax = plt.subplots()

            ax.hist2d(pos[:, 0], pos[:, 1], bins=50)
            ax.set_xlabel('x')
            ax.set_ylabel('y')

            fig.tight_layout()
            plt.show()

    def plot_results(self):

        cnt = np.asarray([len(obj) for obj in self.objects])
        ts = self.tracker.timestamps

        if np.max(cnt) > self.tracker.max_num_objects:
            raise ValueError('More than {} objects detected!'.format(
                    self.max_num_objects))

        fig, axarr = plt.subplots(nrows=1, ncols=1, sharex=True)

        ax = axarr[0]
        ax.set_title('Positions')
        values = np.asarray([obj[0]['pos'] if len(obj) > 0 else (-1, -1)
                             for obj in self.objects])
        ax.plot(ts, values[:, 0], 'b-', label='x')
        ax.plot(ts, values[:, 1], 'r-', label='y')
        ax.set_ylabel('Pixels')

        fig.tight_layout()
        plt.show()

    def save_results(self):

        self.stop()

        # also save bounding box
        bbox = self.tracker.bbox
        bbox_dict = {'x': bbox[0],
                     'y': bbox[1],
                     'w': bbox[2],
                     'h': bbox[3]}

        # create file name from video file path (or use given output folder)
        output = op.split(self.file_path)[0]
        makedirs_save(output)

        filename = op.splitext(op.basename(self.file_path))[0]
        filebase = op.join(output, filename + self.suffix)

        try:
            self.tracker.save_objects(filebase + '.npz',
                                      self.objects,
                                      bbox=bbox_dict)

            self.tracker.save_parameters(
                    filebase + '_tracker_params.npz',
                    automation=self.tracker_parameters,
                    widget_parameters=self.widget_parameters)

        except BaseException:
            traceback.print_exc()

    def close_and_exit(self):

        self.tracker.cleanup()
        self.close()

    def process_all_frames(self):

        n_frames = self.tracker.n_frames
        for i in tqdm.trange(n_frames):
            try:
                objs = self.tracker.process_frame(index=i, draw=False)
                self.objects[i] = objs
            except BaseException:
                self.objects[i] = []

        cnt = np.asarray([len(obj) for obj in self.objects])
        self.plot_item.clear()
        self.plot_item.plot(self.tracker.timestamps, cnt)
