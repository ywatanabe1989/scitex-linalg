"""Smoke tests for vendored decorators.

These mirror the source layout under
``src/scitex_linalg/_vendor_decorators/`` so the package satisfies
``scitex-dev`` rule PS202 (matching tests/ tree). The vendored decorators
are private implementation detail; we only assert they import and expose
the documented callables, so the package free-of-scitex.* runtime deps
guarantee remains testable.
"""

from __future__ import annotations


def test_vendor_decorators_exports() -> None:
    from scitex_linalg import _vendor_decorators as vd

    assert callable(vd.numpy_fn)
    assert callable(vd.torch_fn)
    assert callable(vd.wrap)


def test_wrap_is_identity_preserving() -> None:
    from scitex_linalg._vendor_decorators import wrap

    @wrap
    def add(a: int, b: int) -> int:
        return a + b

    assert add(2, 3) == 5
    assert add.__name__ == "add"


def test_numpy_fn_passthrough_for_ndarray() -> None:
    import numpy as np

    from scitex_linalg._vendor_decorators import numpy_fn

    @numpy_fn
    def double(x):
        return x * 2

    out = double(np.array([1.0, 2.0, 3.0]))
    assert isinstance(out, np.ndarray)
    np.testing.assert_array_equal(out, np.array([2.0, 4.0, 6.0]))
