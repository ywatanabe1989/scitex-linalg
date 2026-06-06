#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Torch tensor numerical helpers.

A small collection of PyTorch tensor utilities migrated from the SciTeX
umbrella (``scitex.torch``):

- :func:`apply_to` — apply a reduction along an arbitrary dim via reshape.
- NaN-aware reductions (``nanmax``, ``nanmin``, ``nanvar``, ``nanstd``,
  ``nanprod``, ``nancumsum``, ``nancumprod``, ``nanargmax``, ``nanargmin``)
  working around https://github.com/pytorch/pytorch/issues/61474.

``torch`` is an optional dependency (``pip install scitex-linalg[torch]``).
The module imports without torch installed; each function raises a clear
:class:`ImportError` on first use when torch is unavailable.
"""

from __future__ import annotations

try:  # torch is an optional extra
    import torch as _torch
except ImportError:  # pragma: no cover — exercised only without torch
    _torch = None


def _require_torch():
    if _torch is None:
        raise ImportError(
            "torch is required for scitex_linalg torch numerics; "
            "install it with `pip install scitex-linalg[torch]`."
        )
    return _torch


def apply_to(fn, x, dim):
    """Apply ``fn`` to each slice of ``x`` along ``dim`` via reshape.

    Parameters
    ----------
    fn : callable
        Reduction applied to each 1-D slice taken along ``dim``.
    x : torch.Tensor
        Input tensor.
    dim : int
        Dimension along which ``fn`` is applied.

    Returns
    -------
    torch.Tensor
        Result with the reduced dimension restored to its original position.

    Example
    -------
    >>> x = torch.randn(2, 3, 4)
    >>> apply_to(sum, x, 1).shape  # (2, 1, 4)
    """
    torch = _require_torch()
    if dim != -1:
        dims = list(range(x.dim()))
        dims[-1], dims[dim] = dims[dim], dims[-1]
        x = x.permute(*dims)

    # Flatten the tensor along the time dimension
    shape = x.shape
    x = x.reshape(-1, shape[-1])

    # Apply the function to each slice along the time dimension
    applied = torch.stack([fn(x_i) for x_i in torch.unbind(x, dim=0)], dim=0)

    # Reshape the tensor to its original shape (with the time dimension at the end)
    applied = applied.reshape(*shape[:-1], -1)

    # Permute back to the original dimension order if necessary
    if dim != -1:
        applied = applied.permute(*dims)

    return applied


# https://github.com/pytorch/pytorch/issues/61474
def nanmax(tensor, dim=None, keepdim=False):
    """NaN-ignoring maximum reduction over ``tensor``."""
    torch = _require_torch()
    min_value = torch.finfo(tensor.dtype).min
    if dim is None:
        output = tensor.nan_to_num(min_value).max()
    else:
        output = tensor.nan_to_num(min_value).max(dim=dim, keepdim=keepdim)
    return output


def nanmin(tensor, dim=None, keepdim=False):
    """NaN-ignoring minimum reduction over ``tensor``."""
    torch = _require_torch()
    max_value = torch.finfo(tensor.dtype).max
    if dim is None:
        output = tensor.nan_to_num(max_value).min()
    else:
        output = tensor.nan_to_num(max_value).min(dim=dim, keepdim=keepdim)
    return output


def nanvar(tensor, dim=None, keepdim=False):
    """NaN-ignoring variance reduction over ``tensor``."""
    _require_torch()
    tensor_mean = tensor.nanmean(dim=dim, keepdim=True)
    output = (tensor - tensor_mean).square().nanmean(dim=dim, keepdim=keepdim)
    return output


def nanstd(tensor, dim=None, keepdim=False):
    """NaN-ignoring standard deviation over ``tensor``."""
    _require_torch()
    output = nanvar(tensor, dim=dim, keepdim=keepdim)
    output = output.sqrt()
    return output


def nanprod(tensor, dim=None, keepdim=False):
    """NaN-ignoring product reduction over ``tensor``."""
    _require_torch()
    if dim is None:
        output = tensor.nan_to_num(1).prod()
    else:
        output = tensor.nan_to_num(1).prod(dim=dim, keepdim=keepdim)
    return output


def nancumprod(tensor, dim=None, keepdim=False):
    """NaN-ignoring cumulative product over ``tensor``."""
    _require_torch()
    if dim is None:
        dim = 0  # Default to first dimension for cumulative operations
    output = tensor.nan_to_num(1).cumprod(dim=dim)
    return output


def nancumsum(tensor, dim=None, keepdim=False):
    """NaN-ignoring cumulative sum over ``tensor``."""
    _require_torch()
    if dim is None:
        dim = 0  # Default to first dimension for cumulative operations
    output = tensor.nan_to_num(0).cumsum(dim=dim)
    return output


def nanargmin(tensor, dim=None, keepdim=False):
    """NaN-ignoring argmin reduction over ``tensor``."""
    torch = _require_torch()
    max_value = torch.finfo(tensor.dtype).max
    if dim is None:
        output = tensor.nan_to_num(max_value).argmin()
    else:
        output = tensor.nan_to_num(max_value).argmin(dim=dim, keepdim=keepdim)
    return output


def nanargmax(tensor, dim=None, keepdim=False):
    """NaN-ignoring argmax reduction over ``tensor``."""
    torch = _require_torch()
    min_value = torch.finfo(tensor.dtype).min
    if dim is None:
        output = tensor.nan_to_num(min_value).argmax()
    else:
        output = tensor.nan_to_num(min_value).argmax(dim=dim, keepdim=keepdim)
    return output


__all__ = [
    "apply_to",
    "nanmax",
    "nanmin",
    "nanvar",
    "nanstd",
    "nanprod",
    "nancumsum",
    "nancumprod",
    "nanargmax",
    "nanargmin",
]
