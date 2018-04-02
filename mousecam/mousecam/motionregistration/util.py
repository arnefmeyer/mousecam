#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Arne F. Meyer <arne.f.meyer@gmail.com>
# License: GPLv3

"""
    some helpers
"""

import numpy as np
import imageio
import cv2


def get_frames(file_path, bbox=None, num_frames=-1,
               brightness_mask=None, offset=0, use_memmap=False):
    """read frames from video file"""

    frames = []
    br_mask = []
    br_bbox = []
    frame_cnt = 0
    rect = bbox

    with imageio.get_reader(file_path, 'ffmpeg') as reader:

        i = 0
        while True:

            try:
                frame = reader.get_data(i)

                if i >= offset:

                    if (i-offset+1) % 1000 == 0:
                        print(str(i-offset+1))

                    if num_frames > 0 and i-offset+1 >= num_frames:
                        break

                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

                    if brightness_mask is None:
                        br_mask.append(np.mean(frame))
                    else:
                        br_mask.append(np.mean(frame[brightness_mask > 0]))

                    if rect is not None:
                        frame = frame[rect[1]:rect[1]+rect[3],
                                      rect[0]:rect[0]+rect[2]]

                    br_bbox.append(np.mean(frame))

                    frames.append(frame)
                    frame_cnt += 1

            except BaseException:
                break

            i += 1

    print rect
    bbox = {'rect': rect,
            'x': rect[0],
            'y': rect[1],
            'w': rect[2],
            'h': rect[3]}

    return np.asarray(frames), bbox, np.asarray(br_bbox), np.asarray(br_mask)


def select_template(frames):

    cv2.namedWindow('frames', cv2.WINDOW_NORMAL)

    def update_frame(index):
        cv2.imshow("frames", frames[index, :, :])

    update_frame(0)
    cv2.createTrackbar('index', 'frames', 0, frames.shape[0]-1, update_frame)

    while True:

        key = cv2.waitKey(100) & 0xFF
        if key in [ord(' '), ord('q')]:
            break

    index = cv2.getTrackbarPos('index', 'frames')

    return frames[index, :, :], index

