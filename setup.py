#!/usr/bin/env python3

from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
    packages=['relformer_transformer'],
    package_dir={'': 'ros'}
)

setup(**d)
