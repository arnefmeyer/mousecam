#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Arne F. Meyer <arne.f.meyer@gmail.com>
# License: GPLv3

"""
    for qt 4/5 compatibility
"""

try:
    from PyQt4 import QtGui, QtCore
    qwidgets = QtGui
except ImportError:
    from PyQt5 import QtGui, QtCore
    import PyQt5.QtWidgets as qwidgets
