# -*- coding: utf-8 -*-

import numpy as np


# define constants
b0 = 1.0 / 1.686

def apply_zreion(density, zmean, alpha, k0, boxsize, Rsmooth=1.0, deconvolve=True):
    """
    Apply zreion ionization history to density field.

    Parameters
    ----------
    density : numpy array
        A numpy array of rank 3 holding the density field.
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

    # make sure that density is actually ovderdensity
    rho_bar = np.mean(density)
    if not np.isclose(rho_bar, 0.0):
        density -= rho_bar
        density /= rho_bar

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
    spherical_k = np.sqrt(kkx**2 + kky**2 + kkz**2)
    bias_val = b0 / (1 + spherical_k / k0)**alpha

    # compute smoothing factor
    smoothing_window = tophat(spherical_k * Rsmooth)

    if deconvolve:
        # compute deconvolution window in grid units
        kkx *= Lx / Nx
        kky *= Ly / Ny
        kkz *= Lz / Nz
        deconv_window = (sinc(kkx / 2) * sinc(kky / 2) * sinc(kkz / 2))**2
    else:
        deconv_window = np.ones_like(density_fft, dtype=np.float64)

    assert bias_val.shape == density_fft.shape, "Bias and density grids are not compatible"
    assert smoothing_window.shape == density_fft.shape, "Smoothing window and density grids are not compatible"
    assert deconv_window.shape == density_fft.shape, "Deconvolution window and density grids are not compatible"

    # apply transformations to Fourier field
    density_fft *= bias_val
    density_fft *= smoothing_window
    density_fft /= deconv_window

    # inverse FFT
    zreion = np.fft.irfftn(density_fft, density.shape)

    # finish computing zreion field
    zreion *= (1 + zmean)
    zreion += zmean

    return zreion


def tophat(x):
    # compute tophat window function
    return np.where(
        np.abs(x) > 1e-6, 3 * (np.sin(x) - x * np.cos(x)) / x**3, 1 - x**2 / 10.0
    )

def sinc(x):
    # compute sinc(x) = sin(x)/x
    return np.where(np.abs(x) > 1e-6, np.sin(x) / x, 1 - x**2 / 6.0)
