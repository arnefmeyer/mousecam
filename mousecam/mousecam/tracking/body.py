#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Arne F. Meyer <arne.f.meyer@gmail.com>
# License: GPLv3

"""
    simple body tracker using opencv's blob detector
"""

from __future__ import print_function

import numpy as np

from base import AbstractTracker, TrackedEllipse, TrackerParameter
from base import AbstractTrackerWidget
from base import InvalidNumberOfObjectsException
from ..util.opencv import cv2, cv


class BodyTracker(AbstractTracker):

    def __init__(self, *args, **kwargs):

        super(BodyTracker, self).__init__(*args, **kwargs)

    def init_parameters(self):

        p = [TrackerParameter('threshold_min', 0, range=(0, 255)),
             TrackerParameter('threshold_max', 100, range=(0, 255)),
             TrackerParameter('mean_pix_max', 90, range=(0, 255)),
             TrackerParameter('area_min', 1, range=(1, 100)),
             TrackerParameter('area_max', 80, range=(1, 100)),
             TrackerParameter('circularity', 10, range=(0, 100)),
             TrackerParameter('convexity', 10, range=(0, 100)),
             TrackerParameter('gaussian_blur', 0, range=(0, 100))]

        return p

    def get_main_clip(self):

        return 'frame'

    def get_extra_clips(self):

        return ['blobs']

    def save_objects(self, filepath, objects, bbox=None):

        cnt = [len(objs) for objs in objects]
        if max(cnt) > 1:
            raise InvalidNumberOfObjectsException(
                "Number of detected objects per frame must not be > 1")

        N = len(objects)
        if N != len(self.timestamps):
            raise ValueError('# of objects != # timestamps')

        center = np.NaN * np.ones((N, 2))
        size = np.NaN * np.ones((N, 2))
        angle = np.NaN * np.ones((N,))
        mean_pix_value = np.NaN * np.ones((N,))
#        timestamp = np.NaN * np.ones((N,))

        for i, objs in enumerate(objects):

            if len(objs) > 0:
                obj = objs[0]
                center[i, :] = obj.pos
                size[i, :] = 2*np.array(obj.axes)
                angle[i] = obj.angle
                mean_pix_value[i] = obj.annotations['mean_pix_value']
#                timestamp[i] = obj.annotations['timestamp']

        print("Saving objects to file:", filepath)
        np.savez(filepath,
                 center=center,
                 size=size,
                 angle=angle,
                 mean_pix_value=mean_pix_value,
                 timestamps=self.timestamps,
                 bbox=bbox)

    def _create_blob_detector(self, frame_size):

        # Setup SimpleBlobDetector parameters.
        params = cv2.SimpleBlobDetector_Params()

        # Change thresholds
        params.minThreshold = self.get_parameter('threshold_min').value
        params.maxThreshold = self.get_parameter('threshold_max').value
        params.thresholdStep = 10

        # Filter by Area.
        area = np.prod(frame_size)
        params.filterByArea = True
        params.minArea = self.get_parameter('area_min').value / 100. * area
        params.maxArea = self.get_parameter('area_max').value / 100. * area

        # Filter by Circularity
        params.filterByCircularity = True
        params.minCircularity = self.get_parameter('circularity').value / 100.

        # Filter by Convexity
        params.filterByConvexity = True
        params.minConvexity = self.get_parameter('convexity').value / 100.

        # Filter by Inertia
        params.filterByInertia = False
        params.minInertiaRatio = 0.01

        params.filterByColor = True
        params.blobColor = 0

        # Create a detector with the parameters
        return cv2.SimpleBlobDetector_create(params)

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

        frame = cv2.cvtColor(frame_gray, cv.CV_GRAY2RGB)

        detector = self._create_blob_detector(frame_gray.shape)

        # gaussian blur
        blur = self.get_parameter('gaussian_blur').value
        if blur > 0:
            if blur % 2 == 0:
                blur += 1
            frame_gray = cv2.GaussianBlur(frame_gray, (blur, blur), 0)

        # blob detection
        # TODO: check parameters if necessary
        keypoints = detector.detect(frame_gray)

        blobs = []
        mean_pix_max = self.get_parameter('mean_pix_max').value
        for kp in keypoints:

            mask = np.zeros_like(frame_gray)
            cv2.ellipse(mask,
                        (int(kp.pt[0]), int(kp.pt[1])),
                        (int(.5*kp.size), int(.5*kp.size)),
                        kp.angle, 0, 360, (255, 255, 255), thickness=-1)
            mean_pix_val = frame_gray[mask > 0].mean()

            if mean_pix_val <= mean_pix_max:
                if self._user_mask is None or self._user_mask[kp.pt[1],
                                                              kp.pt[0]] != 0:
                    blobs.append(TrackedEllipse(kp.pt,
                                                (.5*kp.size, .5*kp.size),
                                                kp.angle,
                                                mean_pix_value=mean_pix_val,
                                                timestamp=self.timestamps[index]))

        if draw:
            frame_blobs = cv2.drawKeypoints(frame, keypoints, np.asarray([]),
                                            (0, 0, 255),
                                            cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

#        if len(keypoints) > 0:
#            print keypoints
#            from ipdb import set_trace as db
#            db()

        if draw:
            return blobs, {'frame': frame,
                           'blobs': frame_blobs}
        else:
            return blobs


class BodyTrackerWidget(AbstractTrackerWidget):

    def __init__(self, file_path, suffix='_body_position', **kwargs):

        super(BodyTrackerWidget, self).__init__(file_path, suffix=suffix,
                                                **kwargs)

    def create_tracker(self, file_path, **kwargs):

        return BodyTracker(file_path, widget=self, **kwargs)
