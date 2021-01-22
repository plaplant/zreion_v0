# -*- coding: utf-8 -*-
"""
    Setup file for zreion.
    Use setup.cfg to configure your project.

    This file was generated with PyScaffold 3.2.3.
    PyScaffold helps you to put up the scaffold of your new Python project.
    Learn more under: https://pyscaffold.org/
"""
import sys

from pkg_resources import VersionConflict, require
from setuptools import setup, Extension

try:
    from Cython.Build import cythonize
except ImportError:

    def cythonize(*args, **kwargs):
        from Cython.Build import cythonize

        return cythonize(*args, **kwargs)


zreion_ext = Extension(
    "zreion._zreion",
    sources=["src/zreion/zreion.pyx"],
    define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
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
