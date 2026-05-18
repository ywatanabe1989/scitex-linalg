"""Smoke tests for vendored decorators.

These mirror the source layout under
``src/scitex_linalg/_vendor_decorators/`` so the package satisfies
``scitex-dev`` rule PS202 (matching tests/ tree). The vendored decorators
are private implementation detail; we only assert they import and expose
the documented callables, so the package free-of-scitex.* runtime deps
guarantee remains testable.
"""

from __future__ import annotations


def test_vendor_decorators_exports_numpy_fn_as_callable() -> None:
    # Arrange
    from scitex_linalg import _vendor_decorators as vd

    # Act
    is_callable = callable(vd.numpy_fn)
    # Assert
    assert is_callable


def test_vendor_decorators_exports_torch_fn_as_callable() -> None:
    # Arrange
    from scitex_linalg import _vendor_decorators as vd

    # Act
    is_callable = callable(vd.torch_fn)
    # Assert
    assert is_callable


def test_vendor_decorators_exports_wrap_as_callable() -> None:
    # Arrange
    from scitex_linalg import _vendor_decorators as vd

    # Act
    is_callable = callable(vd.wrap)
    # Assert
    assert is_callable


def test_wrap_decorator_preserves_function_return_value() -> None:
    # Arrange
    from scitex_linalg._vendor_decorators import wrap

    @wrap
    def add(a: int, b: int) -> int:
        return a + b

    # Act
    result = add(2, 3)
    # Assert
    assert result == 5


def test_wrap_decorator_preserves_function_name_metadata() -> None:
    # Arrange
    from scitex_linalg._vendor_decorators import wrap

    @wrap
    def add(a: int, b: int) -> int:
        return a + b

    # Act
    name = add.__name__
    # Assert
    assert name == "add"


def test_numpy_fn_decorator_returns_numpy_ndarray_for_ndarray_input() -> None:
    # Arrange
    import numpy as np

    from scitex_linalg._vendor_decorators import numpy_fn

    @numpy_fn
    def double(x):
        return x * 2

    # Act
    out = double(np.array([1.0, 2.0, 3.0]))
    # Assert
    assert isinstance(out, np.ndarray)


def test_numpy_fn_decorator_doubles_values_correctly() -> None:
    # Arrange
    import numpy as np

    from scitex_linalg._vendor_decorators import numpy_fn

    @numpy_fn
    def double(x):
        return x * 2

    # Act
    out = double(np.array([1.0, 2.0, 3.0]))
    # Assert
    assert np.array_equal(out, np.array([2.0, 4.0, 6.0]))
