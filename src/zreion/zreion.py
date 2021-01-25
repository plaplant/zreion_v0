# -*- coding: utf-8 -*-

import numpy as np
import pyfftw

from . import _zreion

# define constants
b0 = 1.0 / 1.686


def tophat(x):
    # compute tophat window function
    return np.where(
        np.abs(x) > 1e-6, 3 * (np.sin(x) - x * np.cos(x)) / x ** 3, 1 - x ** 2 / 10.0
    )


def sinc(x):
    # compute sinc(x) = sin(x)/x
    return np.where(np.abs(x) > 1e-6, np.sin(x) / x, 1 - x ** 2 / 6.0)


def _fft3d(array, data_shape, direction="f"):
    """
    Apply an FFT using pyFFTW.

    Parameters
    ----------
    array : ndarray
        The array to apply the transform to.
    data_shape : tuple of int
        The shape of the input data array. Used for determining the size of the
        output array for a forward transform and the target shape for a backward
        one.
    direction : str
        The direction of the transform. Must be either "f" (for forward) or "b"
        (for backward).

    Returns
    -------
    ndarray
        An ndarray of the resulting transform.
    """
    if direction.lower() not in ["f", "b"]:
        raise ValueError(f'"direction" must be "f" or "b", got {direction}')
    dtype = array.dtype.type
    if dtype is np.float32 or dtype is np.complex64:
        # single precision
        input_dtype = "float32"
        output_dtype = "complex64"
    elif dtype is np.float64 or dtype is np.complex128:
        # double precision
        input_dtype = "float64"
        output_dtype = "complex128"
    else:
        raise ValueError("input array must be a real dtype of np.float32 or np.float64")

    if len(data_shape) != 3:
        raise ValueError("input array must be a 3-dimensional array")
    padded_shape = (data_shape[0], data_shape[1], 2 * (data_shape[2] // 2 + 1))
    full_array = pyfftw.empty_aligned(
        padded_shape, input_dtype, n=pyfftw.simd_alignment
    )

    # make input and output arrays
    real_input = full_array[:, :, : data_shape[2]]
    complex_output = full_array.view(output_dtype)

    if direction == "f":
        fftw_obj = pyfftw.FFTW(real_input, complex_output, axes=(0, 1, 2))
        real_input[:] = array
    else:
        fftw_obj = pyfftw.FFTW(
            complex_output, real_input, axes=(0, 1, 2), direction="FFTW_BACKWARD"
        )
        complex_output[:] = array

    # perform computation and return
    return fftw_obj()


def apply_zreion(density, zmean, alpha, k0, boxsize, Rsmooth=1.0, deconvolve=True):
    """
    Apply zreion ionization history to density field.

    Parameters
    ----------
    density : numpy array
        A numpy array of rank 3 holding the density field. Note this should be
        the overdensity field delta = (rho - <rho>)/<rho>, where <rho> is the
        average density value. This quantity has a mean of 0 and a minimum value
        of -1.
    zmean : float
        The mean redshift of reionization.
    alpha : float
        The alpha value of the bias parameter.
    k0 : float
        The k0 value of the bais parameter, in units of h/Mpc.
    boxsize : float or array of floats
        The physical extent of the box along each dimension. If a single value,
        this is assumed to be the same for each axis.
    Rsmooth: float, optional
        The smoothing length of the reionziation field, in Mpc/h. Defaults to 1
        Mpc/h.
    deconvolve : bool, optional
        Whether to deconvolve the CIC particle deposition window. If density
        grid is derived in Eulerian space directly, this step is not needed.

    Returns
    -------
    zreion : numpy array
        A numpy array of rank 3 containing the redshift corresponding to when
        that portion of the volume was reionized.
    """
    # check that parameters make sense
    if len(density.shape) != 3:
        raise ValueError("density must be a 3d array")
    if isinstance(boxsize, (int, float, np.float_)):
        boxsize = np.asarray([np.float64(boxsize)])
    else:
        # assume it's an array of length 1 or 3
        if len(boxsize) not in (1, 3):
            raise ValueError(
                "boxsize must be either a single number or an array of length 3"
            )

    # compute FFT of density field
    density_fft = np.fft.rfftn(density)

    # compute spherical k-indices for cells
    Nx, Ny, Nz = density.shape
    if len(boxsize) == 1:
        Lx = Ly = Lz = boxsize[0]
    else:
        Lx, Ly, Lz = boxsize
    dx = Lx / (2 * np.pi * Nx)
    dy = Ly / (2 * np.pi * Ny)
    dz = Lz / (2 * np.pi * Nz)
    kx = np.fft.fftfreq(Nx, d=dx)
    ky = np.fft.fftfreq(Ny, d=dy)
    kz = np.fft.rfftfreq(Nz, d=dz)  # note we're using rfftfreq!

    # compute bias factor
    kkx, kky, kkz = np.meshgrid(kx, ky, kz)
    spherical_k = np.sqrt(kkx ** 2 + kky ** 2 + kkz ** 2)
    bias_val = b0 / (1 + spherical_k / k0) ** alpha

    # compute smoothing factor
    smoothing_window = tophat(spherical_k * Rsmooth)

    if deconvolve:
        # compute deconvolution window in grid units
        kkx *= Lx / Nx
        kky *= Ly / Ny
        kkz *= Lz / Nz
        deconv_window = (sinc(kkx / 2) * sinc(kky / 2) * sinc(kkz / 2)) ** 2
    else:
        deconv_window = np.ones_like(density_fft, dtype=np.float64)

    assert (
        bias_val.shape == density_fft.shape
    ), "Bias and density grids are not compatible"
    assert (
        smoothing_window.shape == density_fft.shape
    ), "Smoothing window and density grids are not compatible"
    assert (
        deconv_window.shape == density_fft.shape
    ), "Deconvolution window and density grids are not compatible"

    # apply transformations to Fourier field
    density_fft *= bias_val
    density_fft *= smoothing_window
    density_fft /= deconv_window

    # inverse FFT
    zreion = np.fft.irfftn(density_fft, density.shape)

    # finish computing zreion field
    zreion *= 1 + zmean
    zreion += zmean

    return zreion


def apply_zreion_fast(density, zmean, alpha, k0, boxsize, Rsmooth=1.0, deconvolve=True):
    """
    Use as a fast, drop-in replacement for apply_zreion.

    The speedups are accomplished by using pyfftw for FFT computation, and
    parallelized cython code for most of the rest of the calculation. As an
    added benefit, this version requires significantly less memory.

    Parameters
    ----------
    density : numpy array
        A numpy array of rank 3 holding the density field. Note this should be
        the overdensity field delta = (rho - <rho>)/<rho>, where <rho> is the
        average density value. This quantity has a mean of 0 and a minimum value
        of -1.
    zmean : float
        The mean redshift of reionization.
    alpha : float
        The alpha value of the bias parameter.
    k0 : float
        The k0 value of the bais parameter, in units of h/Mpc.
    boxsize : float or array of floats
        The physical extent of the box along each dimension. If a single value,
        this is assumed to be the same for each axis.
    Rsmooth: float, optional
        The smoothing length of the reionziation field, in Mpc/h. Defaults to 1
        Mpc/h.
    deconvolve : bool, optional
        Whether to deconvolve the CIC particle deposition window. If density
        grid is derived in Eulerian space directly, this step is not needed.

    Returns
    -------
    zreion : numpy array
        A numpy array of rank 3 containing the redshift corresponding to when
        that portion of the volume was reionized.
    """
    # check that parameters make sense
    if len(density.shape) != 3:
        raise ValueError("density must be a 3d array")
    if isinstance(boxsize, (int, float, np.float_)):
        boxsize = np.asarray([np.float64(boxsize)])
    else:
        # assume it's an array of length 1 or 3
        if len(boxsize) not in (1, 3):
            raise ValueError(
                "boxsize must be either a single number or an array of length 3"
            )
    # unpack boxsize argument
    if len(boxsize) == 1:
        Lx = Ly = Lz = boxsize[0]
    else:
        Lx, Ly, Lz = boxsize

    # save input array shape
    array_shape = density.shape

    # perform fft
    density_fft = _fft3d(density, array_shape, direction="f")

    # call cython funtion for applying bias relation
    density_fft = np.asarray(
        _zreion._apply_zreion(density_fft, alpha, k0, Lx, Ly, Lz, Rsmooth, deconvolve)
    )

    # perform inverse fft
    density = _fft3d(density_fft, array_shape, direction="b")

    # finish computing zreion field
    density *= 1 + zmean
    density += zmean

    return density
