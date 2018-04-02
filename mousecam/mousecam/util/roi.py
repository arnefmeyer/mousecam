#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Arne F. Meyer <arne.f.meyer@gmail.com>
# License: GPLv3

"""
    ROI selection tools
"""

import numpy as np
import matplotlib.pyplot as plt


def select_ROI_and_mask(frame):

    from matplotlib.widgets import RectangleSelector

    def onselect(eclick, erelease):
        'eclick and erelease are matplotlib events at press and release'
        print(' startposition : (%f, %f)' % (eclick.xdata, eclick.ydata))
        print(' endposition   : (%f, %f)' % (erelease.xdata, erelease.ydata))
        print(' used button   : ', eclick.button)

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    ax1.imshow(frame, vmin=0, vmax=255, cmap='gray')
    img2 = ax2.imshow(frame, vmin=0, vmax=255, cmap='gray')

    current_bbox = []
    current_ellipse = []

    def onselect2(eclick, erelease):
        'eclick and erelease are matplotlib events at press and release'

        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)

        if toggle_selector.RS.active:
            current_bbox[:] = [x1, y1, x2 - x1, y2-y1]

        elif toggle_selector.ES.active:
            xc = x1 + .5*(x2 - x1)
            yc = y1 + .5*(y2 - y1)
            a = .5*abs(x2 - x1)
            b = .5*abs(y2 - y1)

            current_ellipse[:] = [xc, yc, a, b]

        I = np.copy(frame)

        if len(current_ellipse) > 0:
            xc, yc, a, b = current_ellipse
            for ii in range(I.shape[0]):
                for jj in range(I.shape[1]):
                    if (ii - yc) ** 2 / b**2 + (jj - xc)**2 / a**2 > 1:
                        I[ii, jj] = 255

        if len(current_bbox) > 0:
            I = I[y1:y2, x1:x2]

        img2.set_data(I)

    def toggle_selector(event):

        if event.key in ['e', 'E', 'm', 'M']:

            toggle_selector.ES.set_active(True)
            toggle_selector.RS.set_active(False)

        elif event.key in ['r', 'R', 'b', 'B']:

            toggle_selector.ES.set_active(False)
            toggle_selector.RS.set_active(True)

        elif event.key in [' ', 'enter', 'q', 'Q']:
            plt.close()

    toggle_selector.RS = RectangleSelector(ax1, onselect2, drawtype='box')
    toggle_selector.ES = RectangleSelector(ax1, onselect2, drawtype='box')

    toggle_selector.ES.set_active(False)
    toggle_selector.RS.set_active(True)
    plt.connect('key_press_event', toggle_selector)
    plt.show(block=True)

    mask = np.ones(frame.shape, dtype=np.bool)
    if len(current_ellipse) > 0:
        xc, yc, a, b = current_ellipse
        for ii in range(mask.shape[0]):
            for jj in range(mask.shape[1]):
                if (ii - yc) ** 2 / b**2 + (jj - xc)**2 / a**2 > 1:
                    mask[ii, jj] = 0

    return current_bbox, mask
