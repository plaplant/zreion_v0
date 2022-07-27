# -*- coding: utf-8 -*-
# Copyright (c) 2020 Paul La Plante
# Licensed under the MIT License

"""
Setup file for zreion.

Use setup.cfg to configure this project.

This file was generated with PyScaffold 3.2.3.
PyScaffold helps you to put up the scaffold of your new Python project.
Learn more under: https://pyscaffold.org/
"""
import sys

from pkg_resources import VersionConflict, require
from setuptools import setup, Extension

import numpy
from Cython.Build import cythonize


def _is_platform_mac():
    return sys.platform == "darwin"


def _is_platform_windows():
    return sys.platform == "win32"


# define cython compile args, depending on platform
if _is_platform_mac():
    extra_compile_args = ["-O3"]
    extra_link_args = []
elif _is_platform_windows():
    extra_compile_args = ["/Ox", "/openmp"]
    extra_link_args = ["/openmp"]
else:
    extra_compile_args = ["-O3", "-fopenmp"]
    extra_link_args = ["-fopenmp"]

zreion_ext = Extension(
    "zreion._zreion",
    sources=["src/zreion/zreion.pyx"],
    define_macros=[
        ("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION"),
        ("CYTHON_TRACE_NOGIL", "1"),
    ],
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
    include_dirs=[numpy.get_include()],
)

try:
    require("setuptools>=38.3")
except VersionConflict:
    print("Error: version of setuptools is too old (<38.3)!")
    sys.exit(1)


if __name__ == "__main__":
    setup(
        use_pyscaffold=True,
        ext_modules=cythonize([zreion_ext], language_level=3),
    )
