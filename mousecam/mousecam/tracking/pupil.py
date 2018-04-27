#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Arne F. Meyer <arne.f.meyer@gmail.com>
# License: GPLv3

"""
    simple ellipse fitting-based pupil tracker
"""

from __future__ import print_function

import numpy as np
import itertools
from scipy.misc import imresize

from . import inpaintBCT
from .base import AbstractTracker, TrackedEllipse, TrackerParameter
from .base import AbstractTrackerWidget
from .base import InvalidNumberOfObjectsException
from ..util.opencv import cv2, cv


class PupilTracker(AbstractTracker):

    def __init__(self, file_path, max_num_objects=1, **kwargs):

        super(PupilTracker, self).__init__(file_path,
                                           max_num_objects=max_num_objects,
                                           **kwargs)

    def init_parameters(self):

        p = [TrackerParameter('threshold', 90, (0, 255),
                              'Threshold for contour extraction'),
             TrackerParameter('mean_pix_max', 90, (0, 255),
                              'Max. mean pixel intensity of valid ellipses'),
             TrackerParameter('diameter_min', 5, (1, 500),
                              'Minimum ellipse diameter'),
             TrackerParameter('diameter_max', 500, (1, 500),
                              'Maximum ellipse diameter'),
             TrackerParameter('blur1', 0, (0, 100),
                              'Median blur (before blob detection)'),
             TrackerParameter('use_inpainting', 0, (0, 1),
                              'Set to 1 to enable inpainting (0 = disabled)'),
             TrackerParameter('blob_scaling', 150, (0, 200),
                              'Scaling of blob size for inpainting'),
             TrackerParameter('blob_minsize', 1, (0, 100),
                              'Minimum blob size'),
             TrackerParameter('blob_maxsize', 10, (1, 100),
                              'Maximum blob size'),
             TrackerParameter('inpaint_epsilon', 4, (1, 100),
                              'see inpaint algorithm'),
             TrackerParameter('inpaint_kappa', 25, (1, 100),
                              'see inpaint algorithm'),
             TrackerParameter('inpaint_sigma', 2, (1, 25),
                              'see inpaint algorithm'),
             TrackerParameter('inpaint_rho', 3, (1, 25),
                              'see inpaint algorithm'),
             TrackerParameter('inpaint_threshold', 0, (0, 100),
                              'see inpaint algorithm'),
             TrackerParameter('blur2', 0, (0, 100),
                              'Median blur (after inpainting)')]

        return p

    def get_main_clip(self):

        return 'contours'

    def get_extra_clips(self):

        return ['blobs', 'inpaint_mask', 'inpaint']

    def save_objects(self, filepath, objects, bbox=None):

        cnt = [len(objs) for objs in objects]
        if max(cnt) > self.max_num_objects:
            raise InvalidNumberOfObjectsException(
                "Number of detected objects per frame > {}. "
                "Remove objects (e.g., using 'remove mode').".format(
                        self.max_num_objects))

        N = len(objects)
        if N != len(self.timestamps):
            raise ValueError('# of objects != # timestamps')

        center = np.NaN * np.ones((N, 2))
        size = np.NaN * np.ones((N, 2))
        angle = np.NaN * np.ones((N,))
        mean_pix_value = np.NaN * np.ones((N,))

        for i, objs in enumerate(objects):

            if len(objs) > 0:
                obj = objs[0]
                center[i, :] = obj.pos
                size[i, :] = 2*np.array(obj.axes)
                angle[i] = obj.angle
                mean_pix_value[i] = obj.annotations['mean_pix_value']

        print("Saving pupil data to file:", filepath)
        np.savez(filepath,
                 center=center,
                 size=size,
                 angle=angle,
                 mean_pix_value=mean_pix_value,
                 timestamps=self.timestamps,
                 bbox=bbox,
                 oversample=self.oversample)

    def _create_blob_detector(self, minsize=20, maxsize=500):

        # Setup SimpleBlobDetector parameters.
        params = cv2.SimpleBlobDetector_Params()

        # Change thresholds
        params.minThreshold = 100
        params.maxThreshold = 220
        params.thresholdStep = 20

        # Filter by Area.
        params.filterByArea = True
        params.minArea = np.pi*(minsize/2.)**2
        params.maxArea = np.pi*(maxsize/2.)**2

        # Filter by Circularity
        params.filterByCircularity = True
        params.minCircularity = 0.1

        # Filter by Convexity
        params.filterByConvexity = True
        params.minConvexity = 0.5

        # Filter by Inertia
        params.filterByInertia = True
        params.minInertiaRatio = 0.01

        params.filterByColor = True
        params.blobColor = 255

        # Create a detector with the parameters
        ver = (cv2.__version__).split('.')
        if int(ver[0]) < 3:
            detector = cv2.SimpleBlobDetector(params)
        else:
            detector = cv2.SimpleBlobDetector_create(params)

        return detector

    def process_frame(self, index=None, draw=False):

        if index is None:
            index = self.index

        # read frame and convert to gray scale
        frame_gray = self.load_frame(index)
        if frame_gray is None:
            if draw:
                return None, None
            else:
                return None

        oversample = self.oversample
        if oversample is not None and oversample > 1:
            frame_gray = imresize(frame_gray, oversample * 100,
                                  interp='cubic')

        frame = cv2.cvtColor(frame_gray, cv.CV_GRAY2RGB)
        user_mask = self._user_mask

        # median blur before inpainting
        blur = self.get_parameter('blur1').value
        if blur > 0:
            if blur % 2 == 0:
                blur += 1
            frame_gray = cv2.medianBlur(frame_gray, blur)

        frame_inpaint = None
        inpaint_mask = None
        blobs = None
        if self.get_parameter('use_inpainting').value:

            # blob detection
            size = max(frame.shape)
            minsize = self.get_parameter('blob_minsize').value / 100. * size
            maxsize = self.get_parameter('blob_maxsize').value / 100. * size
            detector = self._create_blob_detector(minsize=minsize,
                                                  maxsize=maxsize)
            keypoints = detector.detect(frame_gray)

            if draw:
                blobs = cv2.drawKeypoints(
                        frame,
                        keypoints,
                        np.asarray([]),
                        (0, 0, 255),
                        cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

            scaling = self.get_parameter('blob_scaling').value / 100.

            inpaint_mask = np.zeros_like(frame_gray)

            if scaling > 0 and len(keypoints) > 0:

                for kp in keypoints:
                    cv2.circle(inpaint_mask, (int(kp.pt[0]), int(kp.pt[1])),
                               int(scaling*kp.size),
                               (255, 255, 255), -1)

                # inpaint (note that arrays have to be in fortran order)
                eps = self.get_parameter('inpaint_epsilon').value
                kappa = self.get_parameter('inpaint_kappa').value
                sigma = self.get_parameter('inpaint_sigma').value
                rho = self.get_parameter('inpaint_rho').value
                thresh = self.get_parameter('inpaint_threshold').value

                frame_inpaint = inpaintBCT.inpaintBCT(
                    np.asfortranarray(frame.astype(np.float64)),
                    np.asfortranarray(inpaint_mask.astype(np.float64)),
                    eps, kappa, sigma, rho, thresh)

                frame_inpaint = cv2.cvtColor(frame_inpaint.astype(np.uint8),
                                             cv2.COLOR_RGB2GRAY)
                frame_inpaint = np.ascontiguousarray(frame_inpaint)

        if frame_inpaint is None:
            frame_inpaint = np.copy(frame_gray)

        # ellipse must be inside boundary (if given)
        if user_mask is not None:
            frame_inpaint[user_mask == 0] += 20

        # median blur after inpainting
        blur = self.get_parameter('blur2').value
        if blur > 0:
            if blur % 2 == 0:
                blur += 1
            frame_inpaint = cv2.medianBlur(frame_inpaint, blur)

        # fit ellipses
        threshold = self.get_parameter('threshold').value
        frame_area = np.prod(frame_gray.shape)
        d_min = self.get_parameter('diameter_min').value * oversample / 2.
        d_max = self.get_parameter('diameter_max').value * oversample / 2.
        mean_pix_max = self.get_parameter('mean_pix_max').value

        # thresholding
        status, mask = cv2.threshold(frame_inpaint, threshold, 255,
                                     cv2.THRESH_BINARY)

        # find contours
        try:
            _, contours, hierarchy = cv2.findContours(mask,
                                                      cv2.RETR_TREE,
                                                      cv2.CHAIN_APPROX_NONE)
        except:
            # opencv 2.4.x
            contours, hierarchy = cv2.findContours(mask,
                                                   cv2.RETR_TREE,
                                                   cv2.CHAIN_APPROX_NONE)

        ellipses = []

        if draw:
            frame_ellipses = np.copy(frame)
            frame_contours = np.copy(frame)
            cycler = itertools.cycle([[255, 0, 0],
                                      [0, 255, 0],
                                      [0, 0, 255]])

        for j, contour in enumerate(contours):

            if len(contour) >= 6:

                if draw:
                    cv2.drawContours(frame_contours, contour, -1,
                                     (123, 123, 123), 1)

                center, size, angle = cv2.fitEllipse(contour)

                _center = (int(center[0]), int(center[1]))
                _size = (int(size[0] * 0.5),
                         int(size[1] * 0.5))
                diameter = max(size)

                # get mean pixel intensity of pixels enclosed by ellipse
                msk = np.zeros_like(frame_gray)
                cv2.ellipse(msk, _center, _size,
                            angle, 0, 360,
                            (255, 255, 255), thickness=-1)
                mean_pix_val = frame_inpaint[msk > 0].mean()

                # ellipse area (relative to frame area)
                area = np.prod(_size) * np.pi / frame_area

                inside_bounds = True
                if user_mask is not None:

                        if np.min(center) >= 0 and \
                            _center[0] < frame_gray.shape[0] and \
                                _center[1] < frame_gray.shape[1]:

                            if user_mask[_center[1], _center[0]] == 0:
                                inside_bounds = False

                        else:
                            inside_bounds = False

                if diameter >= d_min and diameter <= d_max and \
                        mean_pix_val <= mean_pix_max and inside_bounds:

                    obj = TrackedEllipse(_center, _size,
                                         angle=angle,
                                         area=area,
                                         mean_pix_value=mean_pix_val,
                                         index=index,
                                         timestamp=self.timestamps[index])
                    if draw:
                        # draw ellipse and ellipse center
                        color = next(cycler)
                        obj.draw(frame_ellipses, draw_center=True, color=color)

                    ellipses.append(obj)

        if draw:
            return ellipses, {'frame': frame,
                              'contours': frame_contours,
                              'ellipses': frame_ellipses,
                              'blobs': blobs,
                              'inpaint_mask': inpaint_mask,
                              'inpaint': frame_inpaint}
        else:
            return ellipses


class PupilTrackerWidget(AbstractTrackerWidget):

    def __init__(self, file_path, suffix='_pupil_data', frame_rate=1,
                 **kwargs):

        super(PupilTrackerWidget, self).__init__(file_path,
                                                 suffix=suffix,
                                                 frame_rate=frame_rate,
                                                 **kwargs)

    def create_tracker(self, file_path, **kwargs):

        return PupilTracker(file_path, widget=self, **kwargs)
