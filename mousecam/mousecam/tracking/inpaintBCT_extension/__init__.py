
from __future__ import print_function

try:
    from . import inpaintBCT
except ImportError:
    print("Could not import inpaintBCT extension.")
    inpaintBCT = None

__all__ = ['inpaintBCT']
