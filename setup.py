#!/usr/bin/env python
import os
from setuptools import setup
import io

def get_version(path):
    with open(path, "r") as f:
        _, version = f.read().strip().split("=")
        version = version.strip().strip('"')
    return version

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

def _read(*parts, **kwargs):
    filepath = os.path.join(os.path.dirname(__file__), *parts)
    encoding = kwargs.pop("encoding", "utf-8")
    with io.open(filepath, encoding=encoding) as fh:
        text = fh.read()
    return text

install_requires = [l for l in _read("requirements.txt").split("\n") if l]

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
      project_urls={
          'Documentation': 'https://adjusttext.readthedocs.io/',
      },
      packages=['adjustText'],
      install_requires=install_requires,
      include_package_data=True,
      long_description=read("README.md"),
      long_description_content_type="text/markdown",
      license="MIT",
      classifiers=[
          'Framework :: Matplotlib',
          'Topic :: Scientific/Engineering :: Visualization'],
     )
