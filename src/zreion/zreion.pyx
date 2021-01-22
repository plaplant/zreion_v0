# -*- mode: python; coding: utf-8 -*-

# distutils: language = c
# cython: linetrace=True
# distutils: define_macros=CYTHON_TRACE_NOGIL=1
# python imports
import numpy as np
import warnings
# cython imports
cimport numpy
cimport cython
from libc.math cimport sin, cos, sqrt, abs, M_PI

# define constants
cdef numpy.float64_t b0_zre = 1.0 / 1.686


# define C-only functions
cdef float tophat(float x):
    if abs(x) > 1e-6:
        return 3 * (sin(x) - x * cos(x)) / x**3
    else:
        return 1 - x**2 / 10

cdef float sinc(float x):
    if abs(x) > 1e-6:
        return sin(x) / x
    else:
        return 1 - x**2 / 6

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef numpy.ndarray[dtype=numpy.float32_t] _apply_zreion(
    numpy.float32_t[:, :, ::1] density,
    numpy.float64_t alpha,
    numpy.float64_t k0,
    numpy.float64_t Lx,
    numpy.float64_t Ly,
    numpy.float64_t Lz,
    numpy.float64_t Rsmooth,
    numpy.npy_bool deconvolve,
):
    cdef Py_ssize_t i,j,k
    cdef float kx,ky,kz,kr,bias
    cdef kx_phys,ky_phys,kz_phys,kr_phys
    cdef float deconv_window,smoothing_window
    cdef unsigned long Nx = density.shape[0]
    cdef unsigned long Ny = density.shape[1]
    cdef unsigned long Nz = density.shape[2]

    for i in range(Nx):
        if i < Nx // 2:
            kx = 2 * M_PI / Nx * i
        else:
            kx = 2 * M_PI / Nx * (i - Nx)
        kx_phys *= Nx / Lx

        for j in range(Ny):
            if j < Ny // 2:
                ky = 2 * M_PI / Ny * j
            else:
                ky = 2 * M_PI / Ny * (j - Nx)
            ky_phys *= Ny / Ly

            for k in range(Nx + 1):
                kz = 2 * M_PI / Nz * k
                kz_phys *= Nz / Lz

                kr = sqrt(kx**2 + ky**2 + kz**2)
                kr_phys = sqrt(kx_phys**2 + ky_phys**2 + kz_phys**2)
                bias = b0_zre / (1 + kr_phys / k0)**alpha
                smoothing_window = tophat(kr_phys * Rsmooth)

                if deconvolve:
                    deconv_window = (sinc(kx / 2) * sinc(ky / 2) * sinc(kz / 2))**2
                else:
                    deconv_window = 1.0

                density[i,j,k] *= bias * smoothing_window / deconv_window

    return density
