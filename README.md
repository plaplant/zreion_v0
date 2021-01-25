# zreion

zreion is a way of quickly computing a "redshift of reionization" field using a
semi-numeric method developed in [Battaglia et
al. (2013)](https://ui.adsabs.harvard.edu/abs/2013ApJ...776...81B/abstract). The
method assumes the redshift of reionization for a particular point in a
cosmological volume is a biased tracer of the matter field, and can be written
as a parameterized bias function. For a full derivation and comparison with
simulations, see the linked paper.

## Installation

Installing the package can be performed by checking the repo out, and then running:

```
pip install .
```

Dependencies should be handled automatically by `pip` if not already installed.

### Installing on macOS

Note that installing on macOS is complicated by the fact that the default
compiler does not support OpenMP, which is used to speed up computation. To
install on macOS, an OpenMP-compatible compiler is required. For example, `gcc`
from [Homebrew](https://brew.sh/) or [MacPorts](https://www.macports.org/)
provides such functionality. Once this has been installed, the appropriate
compiler can be specified using the `CC` variable at build time. For instance,
if using Homebrew to install `gcc-10`, then the package can be built and
installed by running:

```
CC=gcc-10 pip install .
```

Otherwise, the user may get an error such as: `clang: error: unsupported option
'-fopenmp'`.

## Dependencies

* setuptools >= 38.3
* cython
* numpy
* pyfftw

## Tests

The repo includes a test suite, which can be invoked by running `pytest` in the
top-level of the repo. In addition to the package dependencies, it requires
`pytest-cov` and `pytest-cases`. These can be installed using:

```
pip install .[testing]
```
