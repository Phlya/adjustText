#!/usr/bin/env python
import os
from setuptools import setup

def get_version(path):
    with open(path, "r") as f:
        _, version = f.read().strip().split("=")
        version = version.strip().strip('"')
    return version

setup(name='adjustText',
      version=get_version(os.path.join(
          ".", #os.path.dirname(os.path.realpath(__file__)),
          "adjustText",
          "_version.py",
          )),
      description='Iteratively adjust text position in matplotlib plots to minimize overlaps',
      author='Ilya Flyamer',
      author_email='flyamer@gmail.com',
      url='https://github.com/Phlya/adjustText',
      packages=['adjustText'],
      install_requires=['numpy', 'matplotlib'],
      include_package_data=True,

      classifiers=[
          'Framework :: Matplotlib',
          'Topic :: Scientific/Engineering :: Visualization'],
     )
