#!/usr/bin/env python

from setuptools import setup

setup(name='adjustText',
      version='0.6.0',
      description='Iteratively adjust text position in matplotlib plots to minimize overlaps',
      author='Ilya Flyamer',
      author_email='flyamer@gmail.com',
      url='https://github.com/Phlya/adjustText',
      packages=['adjustText'],
      install_requires=['numpy', 'matplotlib']
     )
