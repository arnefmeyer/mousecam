#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Arne F. Meyer <arne.f.meyer@gmail.com>
# License: GPLv3

from __future__ import print_function

import sys
import os
import os.path as op
import shutil
import distutils.command.clean as dcc

DESCRIPTION = 'data extraction/analysis code for the mousecam'
fpath = os.path.split(__file__)[0]
with open(os.path.join(fpath, 'README.md')) as f:
    LONG_DESCRIPTION = f.read()


# Custom clean command to remove build artifacts
class CleanCommand(dcc.clean):
    description = "Remove build artifacts from the source tree"

    def run(self):
        dcc.clean.run(self)

        if os.path.exists('build'):
            shutil.rmtree('build')

        for root, dirs, files in os.walk('mousecam'):

            for f in files:
                if op.splitext(f)[1] in ['.so', '.pyd', '.dll', '.pyc']:
                    os.unlink(op.join(root, f))

            for d in dirs:
                # remove byte code directories
                if d == '__pycache__':
                    shutil.rmtree(os.path.join(root, d))


cmdclass = {'clean': CleanCommand}


def configuration(parent_package='', top_path=None):
    if os.path.exists('MANIFEST'):
        os.remove('MANIFEST')

    from numpy.distutils.misc_util import Configuration
    config = Configuration(None, parent_package, top_path)

    config.set_options(ignore_setup_xxx_py=True,
                       assume_default_configuration=True,
                       delegate_options_to_subpackages=True,
                       quiet=True)

    config.add_subpackage('mousecam')

    return config


def setup_package():

    metadata = dict(name='mousecam',
                    maintainer='Arne F. Meyer',
                    maintainer_email='arne.f.meyer@gmail.com',
                    description=DESCRIPTION,
                    license='GPLv3',
                    url='http://github.com/arnefmeyer/mousecam',
                    version='0.1dev',
                    download_url='http://www.github.com/arnefmeyer/lnpy',
                    long_description=LONG_DESCRIPTION,
                    classifiers=['Intended Audience :: Science/Research',
                                 'License :: GPLv3',
                                 'Programming Language :: C',
                                 'Programming Language :: Python',
                                 'Topic :: Scientific/Neuroscience',
                                 'Operating System :: Unix',
                                 'Programming Language :: Python :: 2.7',
                                 ],
                    cmdclass=cmdclass,
                    packages=['mousecam',
                              'mousecam.io',
                              'mousecam.tracking',
                              'mousecam.movement',
                              'mousecam.util',
                              'mousecam.motionregistration'],)

    if len(sys.argv) >= 2 and \
            ('--help' in sys.argv[1:] or sys.argv[1]
                in ('--help-commands', 'egg_info', '--version', 'clean')):

        try:
            from setuptools import setup
        except ImportError:
            from distutils.core import setup

        metadata['version'] = '0.1dev',
    else:
        from numpy.distutils.core import setup

        metadata['configuration'] = configuration

    try:
        setup(ext_modules=[get_inpaint_extension()],
              **metadata)
    except:
        print(10*"!", "Could not install inpaint cython extension.", 10*"!")
        print(10*"-", "Reason", 10*"-")

        import traceback
        traceback.print_exc()

        print(10*"!", "Installing package without inpaint extension.", 10*"!")
        setup(**metadata)


def build_extensions():

    from numpy.distutils.misc_util import Configuration
    from numpy.distutils.core import setup

    config = Configuration()
    config.add_extension(get_inpaint_extension())

    setup(**config.todict())


def get_inpaint_extension():

    from distutils.extension import Extension
    import numpy

    path = op.join(op.split(op.realpath(__file__))[0],
                   'mousecam',
                   'tracking',
                   'inpaintBCT_extension')

    libraries = []
    if os.name == 'posix':
        libraries.append('m')

    return Extension('mousecam.tracking.inpaintBCT',
                     sources=[op.join(path, 'inpaintBCT.cpp'),
                              op.join(path, 'heap.cpp'),
                              op.join(path, 'inpainting_func.cpp')],
                     libraries=libraries,
                     include_dirs=[numpy.get_include()],
                     extra_compile_args=['-O3'],
                     language='c++')


if __name__ == "__main__":

    # Install or build extension inplace?
    option = 'install'
    if len(sys.argv) > 1:
        option = sys.argv[1]
    print("Building option:", option)

    if option == 'install':
        setup_package()

    elif option == 'build_ext':
        build_extensions()
