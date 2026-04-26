"""Vendored copies of scitex.decorators (numpy_fn / torch_fn / wrap).

Kept here so scitex-linalg has zero scitex.* runtime deps. When scitex-decorators
is extracted as a peer package, this directory should be replaced with a direct
dependency on it.
"""

from ._numpy_fn import numpy_fn
from ._torch_fn import torch_fn
from ._wrap import wrap

__all__ = ["numpy_fn", "torch_fn", "wrap"]
