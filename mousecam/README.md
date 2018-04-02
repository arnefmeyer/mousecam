# mousecam

A Python package containing data extraction and analysis code for the head-mounted camera system for freely moving mice (aka mousecam), e.g.,

* Pupil fitting code (including GUI)
* Body tracking code (including GUI)
* Extraction of dense optical flow (e.g., for whisker pad movement extraction)
* Accelerometer analysis code (e.g., to extract head orientation)


## Dependencies

The package requires a number of other packages, e.g., [opencv](https://picamera.readthedocs.io).


## Installation

1. Install the package via "python setup.py install"
2. For building the Python extensions without installing the package: "python setup.py build_ext -i"

