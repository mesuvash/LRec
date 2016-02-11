#!/usr/bin/env python

# import os
from setuptools import find_packages

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

ext_modules = []
ext_modules += [
    Extension("lrec.evaluate.cy_ranking_metric", [
              "lrec/evaluate/cy_ranking_metric.pyx"]),
    Extension("lrec.utils.data_utils.data_cython_helpers", [
        "lrec/utils/data_utils/data_cython_helpers.pyx"]),
]

setup(
    name="LRec",
    version="0.1",
    author="Suvash Sedhain",
    author_email="mesuvash@gmail.com",
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    package_dir={'': '.'},
    ext_modules=cythonize(ext_modules),
)
