# -*- coding: utf-8 -*-
# Copyright (c) 2020 Paul La Plante
# Licensed under the MIT License

"""Init file for zreion."""
from pkg_resources import get_distribution, DistributionNotFound


try:
    # Change here if project is renamed and does not equal the package name
    dist_name = __name__
    __version__ = get_distribution(dist_name).version
except DistributionNotFound:  # pragma: no cover
    __version__ = "unknown"
finally:
    del get_distribution, DistributionNotFound


from .zreion import apply_zreion, apply_zreion_fast

__all__ = ["apply_zreion", "apply_zreion_fast"]
