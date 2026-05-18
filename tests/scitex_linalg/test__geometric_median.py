#!/usr/bin/env python3
"""TQ-compliant tests for geometric_median.

No mocks — relies on the real `geom_median` library via the
`[torch]` optional extra. Each test: ≥3 word-tokens after `test_`,
single assertion, AAA markers.
"""

import math

import pytest

torch = pytest.importorskip("torch")
geom_median = pytest.importorskip("geom_median")

import numpy as np


# ---------------------------------------------------------------------------
# Basic functionality
# ---------------------------------------------------------------------------


def test_geometric_median_is_callable_from_package():
    # Arrange
    from scitex_linalg import geometric_median

    # Act
    is_callable = callable(geometric_median)
    # Assert
    assert is_callable


def test_geometric_median_1d_per_column_returns_column_medians():
    # Arrange
    from scitex_linalg import geometric_median

    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    # Act
    result = geometric_median(x, dim=-1)
    # Assert
    assert torch.allclose(result, torch.tensor([2.0, 5.0, 8.0]), atol=1e-3)


def test_geometric_median_dim0_returns_centroid_along_first_axis():
    # Arrange
    from scitex_linalg import geometric_median

    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    # Act
    result = geometric_median(x, dim=0)
    # Assert
    assert torch.allclose(result, torch.tensor([4.0, 5.0, 6.0]), atol=1e-3)


def test_geometric_median_3d_input_returns_2d_output_along_last_dim():
    # Arrange
    from scitex_linalg import geometric_median

    torch.manual_seed(42)
    x = torch.randn(2, 3, 4)
    # Act
    result = geometric_median(x, dim=-1)
    # Assert
    assert result.shape == (2, 3)


def test_geometric_median_symmetric_circle_returns_near_origin():
    # Arrange
    from scitex_linalg import geometric_median

    angles = torch.linspace(0, 2 * math.pi, 9, dtype=torch.float32)[:-1]
    points = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)
    # Act
    result = geometric_median(points, dim=0)
    # Assert
    assert torch.allclose(result, torch.zeros(2), atol=1e-3)


def test_geometric_median_single_point_along_dim_returns_that_point():
    # Arrange
    from scitex_linalg import geometric_median

    x = torch.tensor([[1.0], [2.0], [3.0]])
    # Act
    result = geometric_median(x, dim=1)
    # Assert
    assert torch.allclose(result, torch.tensor([1.0, 2.0, 3.0]))


# ---------------------------------------------------------------------------
# Dimension handling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dim", [0, 1, 2])
def test_geometric_median_positive_dim_returns_correct_shape(dim):
    # Arrange
    from scitex_linalg import geometric_median

    torch.manual_seed(0)
    x = torch.randn(3, 4, 5)
    expected = list(x.shape)
    expected.pop(dim)
    # Act
    result = geometric_median(x, dim=dim)
    # Assert
    assert tuple(result.shape) == tuple(expected)


@pytest.mark.parametrize("neg_dim,pos_dim", [(-1, 2), (-2, 1), (-3, 0)])
def test_geometric_median_negative_dim_matches_positive_dim_shape(
    neg_dim, pos_dim
):
    # Arrange
    from scitex_linalg import geometric_median

    torch.manual_seed(1)
    x = torch.randn(3, 4, 5)
    r_pos = geometric_median(x, dim=pos_dim)
    # Act
    r_neg = geometric_median(x, dim=neg_dim)
    # Assert
    assert r_neg.shape == r_pos.shape


def test_geometric_median_dim_out_of_range_raises_index_error():
    # Arrange
    from scitex_linalg import geometric_median

    x = torch.randn(3, 4, 5)
    # Act
    ctx = pytest.raises(IndexError)
    # Assert
    with ctx:
        geometric_median(x, dim=3)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


def test_geometric_median_float32_input_returns_float32_output():
    # Arrange
    from scitex_linalg import geometric_median

    torch.manual_seed(2)
    x = torch.randn(10, 5, dtype=torch.float32)
    # Act
    result = geometric_median(x)
    # Assert
    assert result.dtype == torch.float32


def test_geometric_median_float64_input_returns_float64_output():
    # Arrange
    from scitex_linalg import geometric_median

    torch.manual_seed(3)
    x = torch.randn(10, 5, dtype=torch.float64)
    # Act
    result = geometric_median(x)
    # Assert
    assert result.dtype == torch.float64


def test_geometric_median_requires_grad_input_returns_grad_tensor():
    # Arrange
    from scitex_linalg import geometric_median

    torch.manual_seed(4)
    x = torch.randn(10, 5, requires_grad=True)
    # Act
    result = geometric_median(x)
    # Assert
    assert result.requires_grad


# ---------------------------------------------------------------------------
# @torch_fn decorator — input type handling
# ---------------------------------------------------------------------------


def test_geometric_median_numpy_input_returns_numpy_array():
    # Arrange
    from scitex_linalg import geometric_median

    rng = np.random.default_rng(0)
    x = rng.standard_normal((10, 5))
    # Act
    result = geometric_median(x)
    # Assert
    assert isinstance(result, np.ndarray)


def test_geometric_median_list_input_returns_length_two_result():
    # Arrange
    from scitex_linalg import geometric_median

    x = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    # Act
    result = geometric_median(x)
    # Assert
    assert len(result) == 2


def test_geometric_median_2d_input_returns_1d_output_by_default():
    # Arrange
    from scitex_linalg import geometric_median

    torch.manual_seed(5)
    x = torch.randn(10, 5)
    # Act
    result = geometric_median(x)
    # Assert
    assert result.ndim == 1


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_geometric_median_empty_tensor_raises_index_error():
    # Arrange
    from scitex_linalg import geometric_median

    x = torch.tensor([])
    # Act
    ctx = pytest.raises(IndexError)
    # Assert
    with ctx:
        geometric_median(x)


def test_geometric_median_does_not_mutate_input_tensor():
    # Arrange
    from scitex_linalg import geometric_median

    torch.manual_seed(6)
    x = torch.randn(10, 5)
    snapshot = x.clone()
    # Act
    geometric_median(x)
    # Assert
    assert torch.equal(x, snapshot)


# ---------------------------------------------------------------------------
# Large-scale
# ---------------------------------------------------------------------------


def test_geometric_median_large_tensor_returns_correct_output_shape():
    # Arrange
    from scitex_linalg import geometric_median

    torch.manual_seed(7)
    x = torch.randn(100, 1000)
    # Act
    result = geometric_median(x, dim=1)
    # Assert
    assert result.shape == (100,)


def test_geometric_median_5d_input_along_last_dim_returns_4d_output():
    # Arrange
    from scitex_linalg import geometric_median

    torch.manual_seed(8)
    x = torch.randn(2, 3, 4, 5, 6)
    # Act
    result = geometric_median(x, dim=-1)
    # Assert
    assert result.shape == (2, 3, 4, 5)


# ---------------------------------------------------------------------------
# Integration with real geom_median library
# ---------------------------------------------------------------------------


def test_geometric_median_1d_sample_returns_close_to_median_value():
    # Arrange
    from scitex_linalg import geometric_median

    x = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
    # Act
    result = geometric_median(x, dim=0)
    # Assert
    assert torch.allclose(result, torch.tensor([3.0]), atol=0.1)


# ---------------------------------------------------------------------------
# Signature / API
# ---------------------------------------------------------------------------


def test_geometric_median_signature_exposes_xx_parameter():
    # Arrange
    import inspect

    from scitex_linalg import geometric_median

    # Act
    params = list(inspect.signature(geometric_median).parameters.keys())
    # Assert
    assert "xx" in params


def test_geometric_median_signature_exposes_dim_parameter():
    # Arrange
    import inspect

    from scitex_linalg import geometric_median

    # Act
    params = list(inspect.signature(geometric_median).parameters.keys())
    # Assert
    assert "dim" in params


def test_geometric_median_dim_parameter_defaults_to_negative_one():
    # Arrange
    import inspect

    from scitex_linalg import geometric_median

    # Act
    default = inspect.signature(geometric_median).parameters["dim"].default
    # Assert
    assert default == -1


if __name__ == "__main__":
    import os

    pytest.main([os.path.abspath(__file__)])
