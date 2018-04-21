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
import subprocess

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
                    cmdclass=cmdclass)

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

    setup(**metadata)


def build_extensions():

    # Everything is relative to the path of this script
    path = op.split(op.realpath(__file__))[0]
    dirs = [op.join(path, 'mousecam', 'tracking', 'inpaintBCT')]

    # Make everything Windows compatible
    is_posix = True
    if 'win' in sys.platform:
        is_posix = False

    # Compile Cython code in subpackages
    for d in dirs:

        setup_file = op.join(d, 'setup.py')
        print(setup_file)

        if op.exists(setup_file):

            print("Building:", setup_file)

            if option.lower() == 'build_ext':

                if is_posix:
                    # Just put cython extensions into package subdirectories
                    subprocess.call(["/bin/bash", "-i", "-c",
                                     "{} {} build_ext --inplace".format(
                                             sys.executable, setup_file)])
                else:
                    raise NotImplementedError("compilation of extensions "
                                              "under other than unix not "
                                              "implemented yet")
            else:
                # Install the whole package somewhere
                execfile(setup_file)


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
