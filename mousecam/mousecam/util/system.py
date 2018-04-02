#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Arne F. Meyer <arne.f.meyer@gmail.com>
# License: GPLv3

"""
    filesystem-related functions
"""

import os
import os.path as op


def makedirs_save(d):

    if not op.exists(d):
        try:
            os.makedirs(d)
        except BaseException:
            pass
