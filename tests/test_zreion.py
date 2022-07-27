# -*- coding: utf-8 -*-
# Copyright (c) 2021 Paul La Plante
# Licensed under the MIT License
"""Tests for zreion.zreion module."""

import pytest
import pytest_cases
import numpy as np
from packaging.version import parse

import zreion.zreion as zreion


@pytest.fixture
def fake_data_deterministic():
    """Define a fixture for fake data."""
    is_random = False
    data_shape = (8, 8, 8)
    data = np.empty(data_shape, dtype=np.float32)
    for i in range(data_shape[0]):
        for j in range(data_shape[1]):
            for k in range(data_shape[2]):
                data[i, j, k] = 100 * i + 10 * j + k

    return data, is_random


@pytest.fixture
def fake_data_random():
    """Define a fixture for fake data."""
    is_random = True
    data_shape = (8, 8, 8)
    np.random.seed(8675309)
    data = np.asarray(np.random.normal(size=data_shape), dtype=np.float32)

    return data, is_random


@pytest.fixture
def fake_data_deterministic_anisotropic():
    """Define a fixture for fake data."""
    is_random = False
    data_shape = (13, 11, 9)
    data = np.empty(data_shape, dtype=np.float32)
    for i in range(data_shape[0]):
        for j in range(data_shape[1]):
            for k in range(data_shape[2]):
                data[i, j, k] = i + 1.5 * j - k

    return data, is_random


@pytest.fixture
def fake_data_random_anisotropic():
    """Define a fixture for fake data."""
    is_random = True
    data_shape = (13, 11, 9)
    np.random.seed(8675309)
    data = np.asarray(np.random.normal(size=data_shape), dtype=np.float32)

    return data, is_random


# define cases decorator
fake_data_cases = pytest_cases.parametrize(
    "fake_data",
    [
        pytest_cases.fixture_ref(fake_data_deterministic),
        pytest_cases.fixture_ref(fake_data_random),
        pytest_cases.fixture_ref(fake_data_deterministic_anisotropic),
        pytest_cases.fixture_ref(fake_data_random_anisotropic),
    ],
)


def test_tophat():
    """Test tophat window function calculation."""
    invals = np.linspace(1, 10, num=50)
    refvals = 3 * (np.sin(invals) - invals * np.cos(invals)) / invals**3
    outvals = zreion.tophat(invals)
    assert np.allclose(refvals, outvals)

    # test negative values
    # function is even, so our same refvals work
    invals = -invals
    outvals = zreion.tophat(invals)
    assert np.allclose(refvals, outvals)

    # also test small values
    invals = np.asarray([-1e-7, -1e-8, -1e-9, 0, 1e-9, 1e-8, 1e-7])
    refvals = 1 - invals**2 / 10.0
    # we get a numpy RuntimeWarning with small/invalid values
    with pytest.warns(RuntimeWarning) as record:
        outvals = zreion.tophat(invals)
    assert len(record) == 1
    # error message changes based on numpy version
    np_ver = parse(np.__version__)
    if np_ver.major == 1 and np_ver.minor < 23:
        assert record[0].message.args[0] == "invalid value encountered in true_divide"
    else:
        assert record[0].message.args[0] == "invalid value encountered in divide"

    # make sure values agree
    assert np.allclose(refvals, outvals)

    return


def test_sinc():
    """Test sinc function calculation."""
    invals = np.linspace(1, 10, num=50)
    refvals = np.sin(invals) / invals
    outvals = zreion.sinc(invals)
    assert np.allclose(refvals, outvals)

    # test negative values
    # function is even, so our same refvals work
    invals = -invals
    outvals = zreion.sinc(invals)
    assert np.allclose(refvals, outvals)

    # also test small values
    invals = np.asarray([-1e-7, -1e-8, -1e-9, 0, 1e-9, 1e-8, 1e-7])
    refvals = 1 - invals**2 / 6.0
    # we get a numpy RuntimeWarning with small/invalid values
    with pytest.warns(RuntimeWarning) as record:
        outvals = zreion.sinc(invals)
    assert len(record) == 1
    # error message changes based on numpy version
    np_ver = parse(np.__version__)
    if np_ver.major == 1 and np_ver.minor < 23:
        assert record[0].message.args[0] == "invalid value encountered in true_divide"
    else:
        assert record[0].message.args[0] == "invalid value encountered in divide"

    # make sure values agree
    assert np.allclose(refvals, outvals)

    return


@fake_data_cases
def test_fft3d_forward(fake_data):
    """Test _fft3d function."""
    data, is_random = fake_data

    # use numpy as reference FFT
    numpy_fft = np.fft.rfftn(data)

    # run zreion fft
    input_shape = data.shape
    zreion_fft = zreion._fft3d(data, input_shape, direction="f")
    # need relatively high tolerance because numpy fft is internally done at
    # double precision, seemingly without a way to change it
    assert np.allclose(numpy_fft, zreion_fft, atol=1e-4)
    assert zreion_fft.dtype.type is np.complex64

    return


@fake_data_cases
def test_fft3d_forward_double_precision(fake_data):
    """Test _fft3d function with double-precision."""
    data, is_random = fake_data
    data = np.asarray(data, dtype=np.float64)

    # use numpy as reference FFT
    numpy_fft = np.fft.rfftn(data)

    # run zreion fft
    input_shape = data.shape
    zreion_fft = zreion._fft3d(data, input_shape, direction="f")
    assert np.allclose(numpy_fft, zreion_fft)
    assert zreion_fft.dtype.type is np.complex128

    return


@fake_data_cases
def test_fft3d_backward(fake_data):
    """Test backward _fft3d function."""
    data, is_random = fake_data
    # use numpy as reference FFT
    fake_data_ft = np.asarray(np.fft.rfftn(data), dtype=np.complex64)

    # run zreion fft
    input_shape = data.shape
    zreion_fft = zreion._fft3d(fake_data_ft, input_shape, direction="b")
    assert np.allclose(zreion_fft, data, atol=1e-6)
    assert zreion_fft.dtype.type is np.float32

    return


@fake_data_cases
def test_fft3d_backward_double(fake_data):
    """Test backward _fft3d function with double-precision."""
    data, is_random = fake_data
    data = np.asarray(data, dtype=np.float64)
    fake_data_ft = np.asarray(np.fft.rfftn(data), dtype=np.complex128)

    # run zreion fft
    input_shape = data.shape
    zreion_fft = zreion._fft3d(fake_data_ft, input_shape, direction="b")
    assert np.allclose(zreion_fft, data)
    assert zreion_fft.dtype.type is np.float64

    return


def test_fft3d_direction_error(fake_data_deterministic):
    """Test _fft3d error in direction kwarg."""
    data, is_random = fake_data_deterministic

    # test giving a bogus direction
    with pytest.raises(ValueError) as cm:
        zreion._fft3d(data, data.shape, direction="blah")
    assert str(cm.value).startswith('"direction" must be "f" or "b", got "blah"')

    return


def test_fft3d_datatype_errors(fake_data_deterministic):
    """Test _fft3d errors with bad datatypes."""
    data, is_random = fake_data_deterministic

    # test using incorrect direction for real data
    with pytest.raises(ValueError) as cm:
        zreion._fft3d(data, data.shape, direction="b")
    assert str(cm.value).startswith("a backward transform requires")

    # test using incorrect direction for complex data
    complex_data = np.asarray(data, dtype=np.complex64)
    with pytest.raises(ValueError) as cm:
        zreion._fft3d(complex_data, data.shape, direction="f")
    assert str(cm.value).startswith("a forward transform requires")

    # test using integer datatype
    int_data = np.asarray(data, dtype=np.int32)
    with pytest.raises(ValueError) as cm:
        zreion._fft3d(int_data, data.shape, direction="f")
    assert str(cm.value).startswith("a forward transform requires")

    return


def test_fft3d_rank_errors(fake_data_deterministic):
    """Test _fft3d errors with data rank."""
    data, is_random = fake_data_deterministic
    with pytest.raises(ValueError) as cm:
        zreion._fft3d(data, (0,), direction="f")
    assert str(cm.value).startswith("data_shape must have 3 dimensions")

    data_1d = data[0, 0, :]
    with pytest.raises(ValueError) as cm:
        zreion._fft3d(data_1d, data.shape, direction="f")
    assert str(cm.value).startswith("input array must be a 3-dimensional array")

    return


@fake_data_cases
def test_apply_zreion(fake_data):
    """Test apply_zreion."""
    data, is_random = fake_data
    # clean up input data
    if not is_random:
        data_mean = np.mean(data)
        data = (data - data_mean) / data_mean

    # define model parameters
    zmean = 8.0
    alpha = 0.9
    k0 = 1.0
    boxsize = 10.0
    rsmooth = 1.0
    deconvolve = True
    zre = zreion.apply_zreion(data, zmean, alpha, k0, boxsize, rsmooth, deconvolve)

    # make sure results make sense
    if is_random:
        # agreement is not great for random values because of small array size
        atol = 1e-1
    else:
        atol = 1e-6
    assert np.isclose(np.mean(zre), zmean, atol=atol)

    # also test passing in boxsize as a tuple--should give same answer
    boxsize = (boxsize, boxsize, boxsize)
    zre2 = zreion.apply_zreion(data, zmean, alpha, k0, boxsize, rsmooth, deconvolve)
    assert np.allclose(zre, zre2)

    return


@fake_data_cases
def test_apply_zreion_no_deconvolution(fake_data):
    """Test apply_zreion with no deconvolution."""
    data, is_random = fake_data
    # clean up input data
    if not is_random:
        data_mean = np.mean(data)
        data = (data - data_mean) / data_mean

    # test with no deconvolution
    zmean = 8.0
    alpha = 0.9
    k0 = 1.0
    boxsize = 10.0
    rsmooth = 1.0
    deconvolve = False
    zre = zreion.apply_zreion(data, zmean, alpha, k0, boxsize, rsmooth, deconvolve)

    # make sure results make sense
    if is_random:
        # agreement is not great for random values because of small array size
        atol = 1e-1
    else:
        atol = 1e-6
    assert np.isclose(np.mean(zre), zmean, atol=atol)

    return


def test_apply_zreion_errors(fake_data_deterministic):
    """Test errors in the apply_zreion function."""
    data, is_random = fake_data_deterministic

    # define model parameters
    zmean = 8.0
    alpha = 0.9
    k0 = 1.0
    boxsize = 10.0
    rsmooth = 1.0
    deconvolve = True

    # test using only a 1d density array
    data_1d = data[0, 0, :]
    with pytest.raises(ValueError) as cm:
        zreion.apply_zreion(data_1d, zmean, alpha, k0, boxsize, rsmooth, deconvolve)
    assert str(cm.value).startswith("density must be a 3d array")

    # test using a bad boxsize
    boxsize = (10.0, 10.0)
    with pytest.raises(ValueError) as cm:
        zreion.apply_zreion(data, zmean, alpha, k0, boxsize, rsmooth, deconvolve)
    assert str(cm.value).startswith(
        "boxsize must be either a single number or an array of length 3"
    )

    return


@fake_data_cases
def test_apply_zreion_fast(fake_data):
    """Test apply_zreion_fast."""
    data, is_random = fake_data
    # clean up input data
    if not is_random:
        data_mean = np.mean(data)
        data = (data - data_mean) / data_mean

    # define model parameters
    zmean = 8.0
    alpha = 0.9
    k0 = 1.0
    boxsize = 10.0
    rsmooth = 1.0
    deconvolve = True
    zre = zreion.apply_zreion_fast(data, zmean, alpha, k0, boxsize, rsmooth, deconvolve)

    # make sure results make sense
    if is_random:
        # agreement is not great for random values because of small array size
        atol = 1e-1
    else:
        atol = 1e-6
    assert np.isclose(np.mean(zre), zmean, atol=atol)

    # also test passing in boxsize as a tuple--should give same answer
    boxsize = (boxsize, boxsize, boxsize)
    zre2 = zreion.apply_zreion_fast(
        data, zmean, alpha, k0, boxsize, rsmooth, deconvolve
    )
    assert np.allclose(zre, zre2)

    return


@fake_data_cases
def test_apply_zreion_fast_no_deconvolution(fake_data):
    """Test apply_zreion_fast with no deconvolution."""
    data, is_random = fake_data
    # clean up input data
    if not is_random:
        data_mean = np.mean(data)
        data = (data - data_mean) / data_mean

    # test with no deconvolution
    zmean = 8.0
    alpha = 0.9
    k0 = 1.0
    boxsize = 10.0
    rsmooth = 1.0
    deconvolve = False
    zre = zreion.apply_zreion_fast(data, zmean, alpha, k0, boxsize, rsmooth, deconvolve)

    # make sure results make sense
    if is_random:
        # agreement is not great for random values because of small array size
        atol = 1e-1
    else:
        atol = 1e-6
    assert np.isclose(np.mean(zre), zmean, atol=atol)

    return


def test_apply_zreion_fast_errors(fake_data_deterministic):
    """Test errors in the apply_zreion_fast function."""
    data, is_random = fake_data_deterministic

    # define model parameters
    zmean = 8.0
    alpha = 0.9
    k0 = 1.0
    boxsize = 10.0
    rsmooth = 1.0
    deconvolve = True

    # test using only a 1d density array
    data_1d = data[0, 0, :]
    with pytest.raises(ValueError) as cm:
        zreion.apply_zreion_fast(
            data_1d, zmean, alpha, k0, boxsize, rsmooth, deconvolve
        )
    assert str(cm.value).startswith("density must be a 3d array")

    # test using a bad boxsize
    boxsize = (10.0, 10.0)
    with pytest.raises(ValueError) as cm:
        zreion.apply_zreion_fast(data, zmean, alpha, k0, boxsize, rsmooth, deconvolve)
    assert str(cm.value).startswith(
        "boxsize must be either a single number or an array of length 3"
    )

    return


@fake_data_cases
def test_compare_methods(fake_data):
    """Test that fast and "slow" methods yield same answer."""
    data, is_random = fake_data

    # define model parameters
    zmean = 8.0
    alpha = 0.9
    k0 = 1.0
    boxsize = 10.0
    rsmooth = 1.0
    deconvolve = True

    # use slow function
    zre1 = zreion.apply_zreion(data, zmean, alpha, k0, boxsize, rsmooth, deconvolve)

    # use fast function
    zre2 = zreion.apply_zreion_fast(
        data, zmean, alpha, k0, boxsize, rsmooth, deconvolve
    )

    assert np.allclose(zre1, zre2)

    return
