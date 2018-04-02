#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Arne F. Meyer <arne.f.meyer@gmail.com>
# License: GPLv3

"""
    matplotlib-based GUI to select line in image
"""

from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt


class LineSelector(object):

    def __init__(self, image=None, ax=None, cmap='Greys_r', show=True):

        if ax is None:
            ax = plt.gca()

        self.ax = ax

        if image is not None:
            ax.imshow(image, interpolation='nearest', cmap=cmap)

        self.xy_start = None
        self.xy_stop = None
        self.line = None

        fig = ax.figure
        fig.canvas.mpl_connect('button_press_event', self.press_event)
        fig.canvas.mpl_connect('button_release_event', self.release_event)
        fig.canvas.mpl_connect('key_press_event', self.key_event)

        if show:
            plt.show()

    @property
    def center(self):

        if self.xy_start is None:
            return None
        else:
            dxy = np.asarray(self.xy_stop) - np.asarray(self.xy_start)
            return np.asarray(self.xy_start) + .5*dxy

    @property
    def angle(self):

        if self.xy_start is None:
            return None
        else:
            dxy = np.asarray(self.xy_stop) - np.asarray(self.xy_start)
            return np.arctan(dxy[1] / dxy[0]) / np.pi * 180

    def press_event(self, ev):

        if ev.xdata is not None and ev.ydata is not None:
            self.xy_start = (ev.xdata, ev.ydata)
            print("starting coordinates:", self.xy_start)
            self.xy_stop = None

    def release_event(self, ev):

        if self.xy_start is not None:

            if ev.xdata is not None and ev.ydata is not None:
                self.xy_stop = (ev.xdata, ev.ydata)
                print("stopping coordinates:", self.xy_stop)

                if self.line is not None:
                    self.line.remove()

                x1, y1 = self.xy_start
                x2, y2 = self.xy_stop
                self.line = plt.Line2D([x1, x2], [y1, y2], ls='-', lw=2,
                                       color='r')
                self.ax.add_line(self.line)

                fig = self.ax.figure
                fig.canvas.draw()
                fig.canvas.flush_events()

            else:
                self.xy_start = None
                self.xy_stop = None

        else:
            self.xy_start = None

    def key_event(self, ev):

        if ev.key in [' ', 'q']:
            plt.close(self.ax.figure)
