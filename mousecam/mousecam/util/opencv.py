#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Arne F. Meyer <arne.f.meyer@gmail.com>
# License: GPLv3

"""
    definitions for opencv 2/3 compatility
"""


# workaround to make code work with opencv 2.4.x and 3.x
import cv2
try:
    from cv2 import cv
    CV_VERSION = 2

except:
    cv = cv2
    cv.CV_RETR_LIST = cv.RETR_LIST
    cv.CV_THRESH_BINARY = cv2.THRESH_BINARY
    cv.CV_CHAIN_APPROX_NONE = cv2.CHAIN_APPROX_NONE
    cv.CV_WINDOW_NORMAL = cv2.WINDOW_NORMAL
    cv.CV_RGB2GRAY = cv2.COLOR_RGB2GRAY
    cv.CV_GRAY2RGB = cv2.COLOR_GRAY2RGB
    cv.CreateTrackbar = cv2.createTrackbar
    cv.BackgroundSubtractorMOG = cv2.createBackgroundSubtractorMOG2
    CV_VERSION = 3
