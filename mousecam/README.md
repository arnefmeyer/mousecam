# mousecam

A Python package containing data extraction and analysis code for the head-mounted camera system for freely moving mice (aka mousecam), e.g.,

* Pupil fitting code (including GUI)
* Body tracking code (including GUI)
* Extraction of dense optical flow (e.g., for whisker pad movement)
* Accelerometer analysis code (e.g., to extract head orientation)


## Dependencies

The package requires a number of other packages, e.g., [opencv](https://opencv.org/) (including dependencies) and [pyqtgraph](http://www.pyqtgraph.org/). It is recommended to use the [Anaconda Python distribution](https://www.anaconda.com/download/) and install all packages via conda, e.g., "conda install -c conda-forge opencv" from the command prompt.


## Installation

1. Install the package via "python setup.py install"
2. For building the Python extensions without installing the package: "python setup.py build_ext -i"

**Note:** The package contains a cython extension to wrap the image inpaint C++ code. However, installing extensions on Windows can be a bit tricky (see [here](https://blogs.msdn.microsoft.com/pythonengineering/2016/04/11/unable-to-find-vcvarsall-bat/) for details). If the extension cannot be built then the setup file will install the package without inpaint cython extension. In this case there will be some extra output (check lines with many exclamation marks). 

