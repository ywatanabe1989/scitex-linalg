#!/usr/bin/env python3
"""TQ-compliant tests for torch tensor numerics.

Covers apply_to and the NaN-aware reductions migrated from the
SciTeX umbrella. Each test: >=3 word-tokens after `test_`, single
assertion, AAA markers. No mocks — relies on the real `torch`
library via the `[torch]` optional extra.
"""

import pytest

torch = pytest.importorskip("torch")


# ---------------------------------------------------------------------------
# Package-level import surface
# ---------------------------------------------------------------------------


def test_torch_numerics_names_import_from_package():
    # Arrange
    import scitex_linalg

    names = [
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
    # Act
    all_present = all(hasattr(scitex_linalg, name) for name in names)
    # Assert
    assert all_present


# ---------------------------------------------------------------------------
# apply_to — slice-wise reduction
# ---------------------------------------------------------------------------


def test_apply_to_sum_along_dim_returns_expected_shape():
    # Arrange
    from scitex_linalg import apply_to

    x = torch.randn(2, 3, 4)
    # Act
    result = apply_to(lambda s: s.sum().reshape(1), x, dim=1)
    # Assert
    assert result.shape == (2, 1, 4)


def test_apply_to_last_dim_reduction_matches_manual_sum():
    # Arrange
    from scitex_linalg import apply_to

    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    expected = torch.tensor([[6.0], [15.0]])
    # Act
    result = apply_to(lambda s: s.sum().reshape(1), x, dim=-1)
    # Assert
    assert torch.allclose(result, expected)


# ---------------------------------------------------------------------------
# nanmax — NaN-ignoring maximum
# ---------------------------------------------------------------------------


def test_nanmax_global_ignores_nan_returns_max():
    # Arrange
    from scitex_linalg import nanmax

    tensor = torch.tensor([1.0, float("nan"), 3.0, 2.0])
    # Act
    result = nanmax(tensor)
    # Assert
    assert torch.isclose(result, torch.tensor(3.0))


def test_nanmax_along_dim_returns_per_row_maximum():
    # Arrange
    from scitex_linalg import nanmax

    tensor = torch.tensor([[1.0, float("nan"), 3.0], [float("nan"), 5.0, 4.0]])
    # Act
    values, _indices = nanmax(tensor, dim=1)
    # Assert
    assert torch.allclose(values, torch.tensor([3.0, 5.0]))


# ---------------------------------------------------------------------------
# nanmin — NaN-ignoring minimum
# ---------------------------------------------------------------------------


def test_nanmin_global_ignores_nan_returns_min():
    # Arrange
    from scitex_linalg import nanmin

    tensor = torch.tensor([4.0, float("nan"), 1.0, 2.0])
    # Act
    result = nanmin(tensor)
    # Assert
    assert torch.isclose(result, torch.tensor(1.0))


# ---------------------------------------------------------------------------
# nancumsum — NaN-ignoring cumulative sum
# ---------------------------------------------------------------------------


def test_nancumsum_treats_nan_as_zero_in_cumulative_sum():
    # Arrange
    from scitex_linalg import nancumsum

    tensor = torch.tensor([1.0, float("nan"), 2.0])
    expected = torch.tensor([1.0, 1.0, 3.0])
    # Act
    result = nancumsum(tensor, dim=0)
    # Assert
    assert torch.allclose(result, expected)


if __name__ == "__main__":
    import os

    pytest.main([os.path.abspath(__file__)])
