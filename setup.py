#! /usr/bin/env python
#
# Copyright (C) 2014 Alexandre Gramfort <alexandre.gramfort@inria.fr>

import os

import setuptools  # noqa; analysis:ignore; we are using a setuptools namespace
from numpy.distutils.core import setup

# get the version (don't import mne here, so dependencies are not needed)
version = None
with open(os.path.join('bird', '__init__.py'), 'r') as fid:
    for line in (line.strip() for line in fid):
        if line.startswith('__version__'):
            version = line.split('=')[1].strip().strip('\'')
            break
if version is None:
    raise RuntimeError('Could not determine version')


descr = """BIRD"""

DISTNAME = 'bird'
DESCRIPTION = descr
MAINTAINER = 'Alexandre Gramfort'
MAINTAINER_EMAIL = 'alexandre.gramfort@inria.fr'
URL = 'https://github.com/mmoussallam/bird'
LICENSE = 'BSD (3-clause)'
DOWNLOAD_URL = 'https://github.com/mmoussallam/bird'
VERSION = version


if __name__ == "__main__":
    if os.path.exists('MANIFEST'):
        os.remove('MANIFEST')

    setup(name=DISTNAME,
          maintainer=MAINTAINER,
          include_package_data=True,
          maintainer_email=MAINTAINER_EMAIL,
          description=DESCRIPTION,
          license=LICENSE,
          url=URL,
          version=VERSION,
          download_url=DOWNLOAD_URL,
          python_requires='>=2.7',
          long_description=open('README.rst').read(),
          zip_safe=False,  # the package can run out of an .egg file
          classifiers=['Intended Audience :: Science/Research',
                       'Intended Audience :: Developers',
                       'License :: OSI Approved',
                       'Programming Language :: Python',
                       'Topic :: Software Development',
                       'Topic :: Scientific/Engineering',
                       'Operating System :: Microsoft :: Windows',
                       'Operating System :: POSIX',
                       'Operating System :: Unix',
                       'Operating System :: MacOS'],
          platforms='any',
          packages=['bird', 'bird.tests'],
          )
