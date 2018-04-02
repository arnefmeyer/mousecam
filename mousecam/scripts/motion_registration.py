#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Arne F. Meyer <arne.f.meyer@gmail.com>
# License: GPLv3

"""
    Run motion registration on video file(s)
"""

from __future__ import print_function

import click
import sys
import os
import os.path as op
import glob
import numpy as np
import matplotlib.pyplot as plt

try:
    import mousecam.motionregistration as mmr
    from mousecam.util import selectROI, get_first_frame
    from mousecam.util import plotting as mcp
except ImportError:
    sys.path.append(op.join(op.split(__file__)[0], '..'))
    import mousecam.motionregistration as mmr
    from mousecam.util import selectROI, get_first_frame
    from mousecam.util import plotting as mcp


def find_video_files(path, pattern=None, recursive=False, overwrite=False):

    video_files = []

    if recursive:

        for root, _, _ in os.walk(path, topdown=False):

            files = find_video_files(root, pattern=pattern,
                                     recursive=False)

            if files is not None and len(files) > 0:

                if isinstance(files, list):
                    video_files.extend(files)
                else:
                    video_files.append(files)

    elif op.isfile(path):
        video_files.append(path)

    else:
        if pattern is None:
            pattern = '*.h264'

        files = glob.glob(op.join(path, pattern))
        if not overwrite:
            ff = []
            for f in files:
                mc_file = op.splitext(f)[0] + '_motion_registration.npz'
                if not op.exists(mc_file):
                    print("Adding file:", f)
                    ff.append(f)
                else:
                    print("Skipping file:", f)

            files = ff

        video_files.extend(files)

    return sorted(video_files)


def run_moco(video_path, bbox, offset=0, headless=True, use_average=True,
             max_frames=-1, stddev=3., plot_results=True):

    first_frame = get_first_frame(video_path)

    if bbox is None or len(bbox) == 0:
        bbox = selectROI(first_frame)

    brightness_mask = None  # use rectangular bbox also for brightness

    frames, bbox, br_bbox, br = mmr.get_frames(video_path, bbox,
                                               num_frames=max_frames,
                                               brightness_mask=brightness_mask,
                                               offset=offset)

    if frames is not None:

        # ignore all frames with average brightness exceeding 3 std dev
        yy = np.mean(br_bbox)
        yerr = np.std(br_bbox)
        valid = np.logical_and(br_bbox >= yy-stddev*yerr,
                               br_bbox <= yy+stddev*yerr)
        template = np.mean(frames[valid, :, :], axis=0)

        # run moco using wrapper class
        moco = mmr.Moco()
        xy_shift = moco.run(frames,
                            headless=headless,
                            template=template,
                            use_average=use_average)

        result_file = op.splitext(video_path)[0] + '_motion_registration.npz'
        np.savez(result_file,
                 video_path=video_path,
                 xy_shift=xy_shift,
                 bbox=bbox,
                 brightness_bbox=br_bbox,
                 brightness_frames=br,
                 brightness_mask=brightness_mask,
                 template=template)

        mean_mvmt = np.mean(np.abs(np.diff(xy_shift, axis=0)), axis=0)
        print("average movement x/y:", mean_mvmt[0], mean_mvmt[1])

        if plot_results:

            fig, axarr = plt.subplots(nrows=2, ncols=1, sharex=True)
            xx = 1 + np.arange(xy_shift.shape[0])

            red = mcp.NICE_COLORS['new mpl red']
            blue = mcp.NICE_COLORS['new mpl blue']

            ax = axarr[0]
            ax.plot(xx, xy_shift[:, 0], '-', color=red, lw=2, label='x')
            ax.plot(xx, xy_shift[:, 1], '-', color=blue, lw=2, label='y')
            ax.set_xlabel('Frame index')
            ax.set_ylabel('Shift (pixels)')

            ax.set_ylim(np.min(xy_shift)-1, np.max(xy_shift)+1)
            ax.legend(loc='best', fontsize=8)

            ax = axarr[1]
            ax.plot(xx, br_bbox, '-', color=red, lw=2)

            ax.axhline(yy, color=blue)
            ax.axhline(yy - 2*yerr, color=blue, ls='--')
            ax.axhline(yy + 2*yerr, color=blue, ls='--')
            ax.set_xlabel('Frame index')
            ax.set_ylabel('Mean brightness')

            for ax in axarr.flat:

                mcp.set_font_axes(ax, add_size=2)
                mcp.simple_xy_axes(ax)

                ax.xaxis.set_major_locator(plt.MaxNLocator(4))
                ax.yaxis.set_major_locator(plt.MaxNLocator(4))

            fig.set_size_inches(7, 3)
            fig.tight_layout()


@click.command()
@click.argument('path', type=click.Path(exists=True))
@click.option('--bbox', '-b', nargs=4, type=int, default=None)
@click.option('--frames', '-f', default=-1, type=int)
@click.option('--offset', '-o', default=0, type=int)
@click.option('--overwrite', '-w', is_flag=True)
@click.option('--headless', '-H', is_flag=True)
@click.option('--average', '-a', is_flag=True)
@click.option('--recursive', '-r', is_flag=True)
@click.option('--pattern', '-p', default='*.h264', type=str)
@click.option('--batch', '-B', is_flag=True)
@click.option('--show', '-s', is_flag=True)
def cli(path=None, bbox=None, frames=-1, overwrite=False,
        headless=False, average=False, offset=0,
        recursive=False, pattern=None, batch=False,
        show=False):

    video_files = find_video_files(path, pattern, recursive,
                                   overwrite=overwrite)

    if len(video_files) > 0:

        for f in video_files:
            print("Found file:", f)

        bboxes = []
        if bbox is None or len(bbox) == 0:

            if batch:

                # select ROIs (bounding boxes) for all video files
                for video_file in video_files:

                    first_frame = get_first_frame(video_file)
                    bbox = selectROI(first_frame, title=video_file)
                    bboxes.append(bbox)

            else:
                first_frame = get_first_frame(video_files[0])
                bbox = selectROI(first_frame)
                bboxes = len(video_files) * [bbox]
        else:
            bboxes = len(video_files) * [bbox]

        for bbox, video_file in zip(bboxes, video_files):

            if bbox is not None:
                run_moco(video_file, bbox,
                         offset=offset,
                         use_average=average,
                         headless=headless,
                         max_frames=frames,
                         plot_results=show)

            else:
                print("Skipping video file:", video_file)

    if show:
        plt.show()


if __name__ == '__main__':
    cli()
