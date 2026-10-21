# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib


from os.path import dirname, realpath
import os
from setuptools import setup, find_packages, Distribution
from opencood.version import __version__


def _read_requirements_file():
    """Return the elements in requirements.txt."""
    req_file_path = os.path.join(dirname(realpath(__file__)),
                                 os.environ.get('CSCWD_REQUIREMENTS', 'requirements.txt'))
    with open(req_file_path) as f:
        return [line.strip() for line in f
                if line.strip() and not line.lstrip().startswith('#')]


setup(
    name='OpenCOOD',
    version=__version__,
    packages=find_packages(),
    url='https://github.com/ucla-mobility/OpenCDA.git',
    license='MIT',
    author='Runsheng Xu, Hao Xiang',
    author_email='rxx3386@ucla.edu',
    description='An opensource pytorch framework for autonomous driving '
                'cooperative detection',
    long_description=open("readme.md").read(),
    install_requires=_read_requirements_file(),
)
