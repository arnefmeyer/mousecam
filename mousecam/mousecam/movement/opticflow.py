#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Arne F. Meyer <arne.f.meyer@gmail.com>
# License: GPLv3

"""
    optical flow-related functions
"""

import numpy as np
import cv2


def compute_flow_dense(video_file, n_frames, bbox=None, sharpen=False,
                       func=None):
    """compute dense optical flow for video using Farneback's algorithm

        http://docs.opencv.org/trunk/d7/d8b/tutorial_py_lucas_kanade.html
    """

    import imageio
    import tqdm

    flow_mag = np.zeros((n_frames, 2))
    brightness = np.zeros((n_frames,))

    if sharpen:
        # can be helpful for blurry images
        kernel_sharpen = np.array([[-1, -1, -1],
                                   [-1, 9, -1],
                                   [-1, -1, -1]])

    with imageio.get_reader(video_file, 'ffmpeg') as reader:

        flow_xy = np.zeros((n_frames, 2))
        flow_mag = np.zeros((n_frames,))

        previous_frame = None
        pbar = tqdm.tqdm(total=n_frames)
        for i, frame in enumerate(reader):

            pbar.update(1+i)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

            if bbox is not None:
                frame = frame[bbox[1]:bbox[1]+bbox[3],
                              bbox[0]:bbox[0]+bbox[2]]

            if sharpen:
                frame = cv2.filter2D(frame, -1, kernel_sharpen)

            if func is not None:
                frame = func(frame)

            if i > 0:

                if int(cv2.__version__[0]) == 2:
                    flow = cv2.calcOpticalFlowFarneback(previous_frame, frame,
                                                        0.5, 3, 15, 3, 5,
                                                        1.2, 0)
                else:
                    flow = cv2.calcOpticalFlowFarneback(previous_frame, frame,
                                                        None, 0.5, 3, 15, 3,
                                                        5, 1.2, 0)

                flow_xy[i, :] = [np.mean(flow[:, :, 0]),
                                 np.mean(flow[:, :, 1])]
                flow_mag[i] = np.mean(np.sqrt(np.sum(flow**2, axis=2)))

                brightness[i] = np.mean(frame)

            previous_frame = frame

            if i + 1 >= n_frames:
                break

        pbar.close()

    return flow_xy, flow_mag, brightness


def draw_flow_field(img, flow, step=16):
    """adapted from opencv's opt_flow.py"""

    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)

    vis = np.copy(img)
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (x2, y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)

    return vis


def draw_flow_direction(img, flow, scale=1):
    """draw net x/y flow direction"""

    center = (np.asarray(img.shape[:2])[::-1] / 2.).astype(np.int)
    xy = np.array([np.mean(flow[:, :, 0]),
                   np.mean(flow[:, :, 1])])

    vis = np.copy(img)
    cv2.line(vis, tuple(center),
             tuple((center+scale*xy).astype(np.int)), [0, 0, 255],
             thickness=2, lineType=8, shift=0)

    return vis
