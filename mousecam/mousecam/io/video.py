#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Arne F. Meyer <arne.f.meyer@gmail.com>
# License: GPLv3

from __future__ import print_function

import os
import os.path as op
import numpy as np

from ..util.opencv import cv2
from ..util.opencv import cv
from ..util.roi import select_ROI_and_mask


def get_video_files(path, extensions=['.mp4']):

    video_files = []

    if op.isfile(path) and op.splitext(path)[1] in extensions:
        video_files = [path]

    elif op.isdir(path):

        video_files = []
        for root, dirs, files in os.walk(path):

            for f in files:

                if op.splitext(f)[1] in extensions:
                    video_files.append(op.join(root, f))

    return video_files


def get_first_frame(file_path,
                    grayscale=True):

    import imageio

    with imageio.get_reader(file_path, 'ffmpeg') as reader:

        frame = reader.get_data(0)

        if grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    return frame


def dump_video_to_memmap_file(file_path,
                              output=None,
                              overwrite=False,
                              bbox=None,
                              max_num_frames=-1,
                              timestamps=None):

    import tqdm
    import imageio

    # memmap file path
    if output is None:
        filebase = op.splitext(file_path)[0]
    else:
        filebase = op.join(output,
                           op.splitext(op.basename(file_path))[0])

    mmap_file = filebase + '_memmap.bin'
    param_file = filebase + '_memmap_params.npy'

    if not op.exists(mmap_file) or overwrite:

        # open reader and get video parameters
        reader = imageio.get_reader(file_path, 'ffmpeg')
        meta = reader.get_meta_data()
        w, h = meta['size']

        n_frames = meta['nframes']
        if np.isinf(n_frames):
            # some version return inf
            import cv2
            cap = cv2.VideoCapture(file_path)
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if max_num_frames > 0:
            n_frames = min(max_num_frames, n_frames)

        if timestamps is None:
            timestamps = np.arange(n_frames)
        else:
            timestamps = timestamps[:n_frames]

        # process frames
        fp = None
        size = None
        mask = None

        for i in tqdm.trange(n_frames):

            frame = cv2.cvtColor(reader.get_data(i), cv.CV_RGB2GRAY)

            if i == 0:

                if bbox is None or len(bbox) == 0:

                    bbox, mask = select_ROI_and_mask(frame)
                    if len(bbox) == 0:
                        bbox = [0, 0, frame.shape[1], frame.shape[0]]

                print("bounding box:", bbox)

                size = (n_frames, bbox[3], bbox[2])
                fp = np.memmap(mmap_file, dtype='uint8', mode='w+', shape=size)

            if mask is not None:
                frame[~mask] = 255

            if len(bbox) > 0:
                frame = frame[bbox[1]:bbox[1]+bbox[3],
                              bbox[0]:bbox[0]+bbox[2]]

            fp[i, :, :] = frame

        # make sure to flush file
        del fp

        # save parameters to numpy file
        dd = {'file': mmap_file,
              'bbox': bbox,
              'n_frames': n_frames,
              'original_width': w,
              'original_height': h,
              'width': bbox[2],
              'height': bbox[3],
              'w_offset': bbox[0],
              'h_offset': bbox[1],
              'dtype': 'uint8',
              'size': size,
              'timestamps': timestamps}

        np.save(param_file, dd)

    else:
        dd = np.load(param_file).item()

    return dd
