#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Arne F. Meyer <arne.f.meyer@gmail.com>
# License: GPLv3

"""
    Image registration using the moco algorithm

    For details see
    https://www.frontiersin.org/articles/10.3389/fninf.2016.00006/full

    An imagej plugin is available here:
    https://github.com/NTCColumbia/moco

    The wrapper code assumes that fiji (a version of imagej with batteries
    included, https://fiji.sc/) is installed on the system and callable via
    "fiji" from the command line.
"""

from __future__ import print_function

import tempfile
import shutil
import os.path as op
import numpy as np
import cv2
import sys
import subprocess

from .util import select_template


class Moco(object):
    """Wrapper class for the moco plugin"""

    def __init__(self, temp_path=None, max_motion=1000):

        self.temp_path = temp_path
        self.max_motion = max_motion

    def run(self, frames, headless=True,
            template=None, use_average=True,
            verbose=True):

        # create temporary directory
        temp_path = self.temp_path
        rm_path = False
        if temp_path is None:
            temp_path = tempfile.mkdtemp('moco')
            rm_path = True

        print("temp path:", temp_path)

        if isinstance(frames, (np.ndarray)):
            # write frames to binary file
            stack_file = op.join(temp_path, 'frames.raw')
            frames.tofile(stack_file, format='%d')
        elif isinstance(frames, str) and op.exists(frames):
            stack_file = frames

        # template frame
        if template is None:

            if use_average:
                if verbose:
                    print("Using average frame as template")
                template = np.mean(frames, axis=0)
            else:
                template, index = select_template(frames)

        template_file = op.join(temp_path, 'template.png')
        print("Template file:", template_file)
        cv2.imwrite(template_file, template)

        # create imagej macro script file
        macro_file, results_file = self.create_macro_script(stack_file,
                                                            frames.shape,
                                                            template_file)

        self._call_moco(macro_file, headless=headless)

        values = np.genfromtxt(results_file, dtype=np.int,
                               skip_header=1, delimiter=',')

        if rm_path:
            shutil.rmtree(temp_path)

        return values[:, 1:]

    def create_macro_script(self, stack_file, shape, template_file):

        maxval = self.max_motion
        macro_file = op.join(op.split(stack_file)[0], 'macro.ijm')
        results_file = op.join(op.split(stack_file)[0], 'results.csv')

        with open(macro_file, 'w') as f:

            f.write('run("Raw...", "open={} image=8-bit width={} height={} number={}");\n'.format(
                stack_file, shape[2], shape[1], shape[0]))

            f.write('open("{}");\n'.format(template_file))

            f.write('run("moco ", "value={} downsample_value=0 template={} stack={} log=[Generate log file] plot=[No plot]");\n'.format(
                maxval, op.split(template_file)[1], op.split(stack_file)[1]))

            f.write('saveAs("Results", "{}");\n'.format(results_file))

            f.write('run("Quit");\n')

        return macro_file, results_file

    def _call_moco(self, macro_file, headless=True):

        cmd_list = ['fiji', '-macro', macro_file]
        if headless:
            cmd_list.append(' --headless')

        try:
            cmd = ' '.join(cmd_list)
            retcode = subprocess.call(cmd, shell=True)
            if retcode < 0:
                print("Child was terminated by signal", -retcode,
                      file=sys.stderr)
            else:
                print("Child returned", retcode, file=sys.stderr)
        except OSError as e:
            print("Execution failed:", e, file=sys.stderr)

