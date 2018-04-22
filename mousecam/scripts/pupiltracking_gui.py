#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Arne F. Meyer <arne.f.meyer@gmail.com>
# License: GPLv3

"""
    simple opencv and qt-based pupil tracking widget

    NOTE: currently only mp4 files are supported. However, on Linux h264 files
    (e.g., recorded using a Raspberry Pi camera) can easily be convert to mp4
    using MP4Box (GPAC, https://gpac.wp.imt.fr/mp4box/):

        MP4Box -fps RATE -add H264_PATH MP4_FILE_NAME.mp4

    RATE: the frame rate of the h264 file
    H264_PATH: location of the file
    MP4_FILE_NAME: output file.name.

"""

from __future__ import print_function

import click

from mousecam.util.pyqt import qwidgets as qw
from mousecam.io.video import get_video_files
from mousecam.tracking.pupil import PupilTrackerWidget


@click.command()
@click.argument('path', type=click.Path(exists=True))
@click.option('--frames', '-f', default=-1, type=int)
@click.option('--parameters', '-p', default=None,
              type=click.Path(exists=True))
@click.option('--overwrite', '-w', is_flag=True)
@click.option('--bbox', '-b', nargs=4, type=int, default=None)
@click.option('--output', '-o', default=None)
@click.option('--suffix', '-s', default='_pupil_data')
@click.option('--oversample', '-O', default=1)
def cli(path=None, frames=None, parameters=None, **kwargs):

    app = qw.QApplication([])

    video_files = get_video_files(path)

    for video_file in video_files:
        w = PupilTrackerWidget(video_file,
                               max_num_frames=frames,
                               param_file=parameters,
                               **kwargs)
        w.show()
        app.exec_()


if __name__ == '__main__':
    cli()
