# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2020 Paul La Plante
# Licensed under the MIT License

# distutils: language = c
# cython: linetrace=True
# python imports
import numpy as np
import warnings
from cython.parallel import prange
# cython imports
cimport numpy
cimport cython
from libc.math cimport sin, cos, sqrt, fabs, M_PI

# define constants
cdef numpy.float64_t b0_zre = 1.0 / 1.686


# define C-only functions
@cython.cdivision(True)
cdef double tophat(double x) nogil:
    if fabs(x) > 1e-6:
        return 3 * (sin(x) - x * cos(x)) / x**3
    else:
        return 1 - x**2 / 10

@cython.cdivision(True)
cdef double sinc(double x) nogil:
    if fabs(x) > 1e-6:
        return sin(x) / x
    else:
        return 1 - x**2 / 6

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef numpy.complex64_t[:, :, :] _apply_zreion(
    numpy.complex64_t[:, :, :] density,
    numpy.float64_t alpha,
    numpy.float64_t k0,
    numpy.float64_t Lx,
    numpy.float64_t Ly,
    numpy.float64_t Lz,
    numpy.float64_t Rsmooth,
    numpy.npy_bool deconvolve,
    numpy.npy_bool odd_Nz,
):
    cdef int i,j,k
    cdef double kx,ky,kz,bias
    cdef double kx_phys,ky_phys,kz_phys,kr_phys
    cdef double deconv_window,smoothing_window
    cdef int Nx = density.shape[0]
    cdef int Ny = density.shape[1]
    cdef int Nz = density.shape[2]
    cdef int Nz_full = 2 * (Nz - 1)
    if odd_Nz:
        Nz_full += 1

    for i in prange(Nx, nogil=True):
        if i < (Nx // 2) + 1:
            kx = 2 * M_PI / Nx * i
        else:
            kx = 2 * M_PI / Nx * (i - Nx)
        kx_phys = kx * (Nx / Lx)

        for j in range(Ny):
            if j < (Ny // 2) + 1:
                ky = 2 * M_PI / Ny * j
            else:
                ky = 2 * M_PI / Ny * (j - Ny)
            ky_phys = ky * (Ny / Ly)

            for k in range(Nz):
                kz = 2 * M_PI / Nz_full * k
                kz_phys = kz * (Nz_full / Lz)

                kr_phys = sqrt(kx_phys**2 + ky_phys**2 + kz_phys**2)
                bias = b0_zre / (1 + kr_phys / k0)**alpha
                smoothing_window = tophat(kr_phys * Rsmooth)

                if deconvolve:
                    deconv_window = (sinc(kx / 2) * sinc(ky / 2) * sinc(kz / 2))**2
                else:
                    deconv_window = 1.0

                # in-place multiplication causes compile error because of complex type
                density[i,j,k] = density[i,j,k] * (bias * smoothing_window / deconv_window)

    return density
