#!/usr/bin/env python3
"""TQ-compliant tests for distance computation functions.

Covers euclidean_distance, cdist wrapper, and edist alias.
Each test: ≥3 word-tokens after `test_`, single assertion, AAA markers.
"""

import pytest

scipy = pytest.importorskip("scipy")
torch = pytest.importorskip("torch")

import numpy as np
import scipy.spatial.distance as scipy_distance


# ---------------------------------------------------------------------------
# euclidean_distance — basic correctness
# ---------------------------------------------------------------------------


def test_euclidean_distance_returns_sqrt3_for_unit_offsets_1d():
    # Arrange
    from scitex_linalg import euclidean_distance

    uu = np.array([0, 0, 0])
    vv = np.array([1, 1, 1])
    # Act
    dist = euclidean_distance(uu, vv, axis=0)
    # Assert
    assert np.allclose(dist, np.sqrt(3))


def test_euclidean_distance_2d_first_pair_matches_manual_calculation():
    # Arrange
    from scitex_linalg import euclidean_distance

    uu = np.array([[0, 0], [1, 1], [2, 2]])
    vv = np.array([[3, 3], [4, 4], [5, 5]])
    expected = np.sqrt((3 - 0) ** 2 + (4 - 1) ** 2 + (5 - 2) ** 2)
    # Act
    dist = euclidean_distance(uu, vv, axis=0)
    # Assert
    assert np.isclose(dist[0, 0], expected)


def test_euclidean_distance_3d_axis0_returns_expected_shape():
    # Arrange
    from scitex_linalg import euclidean_distance

    uu = np.random.rand(4, 3, 5)
    vv = np.random.rand(4, 3, 5)
    # Act
    dist = euclidean_distance(uu, vv, axis=0)
    # Assert
    assert dist.shape == (3, 5, 3, 5)


def test_euclidean_distance_3d_axis1_returns_expected_shape():
    # Arrange
    from scitex_linalg import euclidean_distance

    uu = np.random.rand(4, 3, 5)
    vv = np.random.rand(4, 3, 5)
    # Act
    dist = euclidean_distance(uu, vv, axis=1)
    # Assert
    assert dist.shape == (3, 5, 4, 5)


def test_euclidean_distance_3d_axis2_returns_expected_shape():
    # Arrange
    from scitex_linalg import euclidean_distance

    uu = np.random.rand(4, 3, 5)
    vv = np.random.rand(4, 3, 5)
    # Act
    dist = euclidean_distance(uu, vv, axis=2)
    # Assert
    assert dist.shape == (5, 4, 4, 3)


def test_euclidean_distance_scalar_inputs_returns_absolute_difference():
    # Arrange
    from scitex_linalg import euclidean_distance

    uu = 3.0
    vv = 7.0
    # Act
    dist = euclidean_distance(uu, vv)
    # Assert
    assert np.allclose(dist, 4.0)


def test_euclidean_distance_identical_arrays_yield_zero_diagonal():
    # Arrange
    from scitex_linalg import euclidean_distance

    arr = np.random.rand(5, 3)
    # Act
    dist = euclidean_distance(arr, arr, axis=0)
    # Assert
    assert np.allclose(np.diag(dist), 0)


# ---------------------------------------------------------------------------
# euclidean_distance — axis parameter
# ---------------------------------------------------------------------------


def test_euclidean_distance_2d_axis0_returns_3_by_3_matrix():
    # Arrange
    from scitex_linalg import euclidean_distance

    uu = np.array([[1, 2, 3], [4, 5, 6]])
    vv = np.array([[7, 8, 9], [10, 11, 12]])
    # Act
    dist = euclidean_distance(uu, vv, axis=0)
    # Assert
    assert dist.shape == (3, 3)


def test_euclidean_distance_2d_axis1_returns_3_by_2_matrix():
    # Arrange
    from scitex_linalg import euclidean_distance

    uu = np.array([[1, 2, 3], [4, 5, 6]])
    vv = np.array([[7, 8, 9], [10, 11, 12]])
    # Act
    dist = euclidean_distance(uu, vv, axis=1)
    # Assert
    assert dist.shape == (3, 2)


def test_euclidean_distance_negative_axis_minus_one_returns_4d():
    # Arrange
    from scitex_linalg import euclidean_distance

    uu = np.random.rand(3, 4, 5)
    vv = np.random.rand(3, 4, 5)
    # Act
    dist = euclidean_distance(uu, vv, axis=-1)
    # Assert
    assert dist.ndim == 4


def test_euclidean_distance_negative_axis_yields_no_nan_values():
    # Arrange
    from scitex_linalg import euclidean_distance

    uu = np.random.rand(3, 4, 5)
    vv = np.random.rand(3, 4, 5)
    # Act
    dist = euclidean_distance(uu, vv, axis=-1)
    # Assert
    assert not np.any(np.isnan(dist))


def test_euclidean_distance_negative_axis_yields_non_negative_values():
    # Arrange
    from scitex_linalg import euclidean_distance

    uu = np.random.rand(3, 4, 5)
    vv = np.random.rand(3, 4, 5)
    # Act
    dist = euclidean_distance(uu, vv, axis=-1)
    # Assert
    assert np.all(dist >= 0)


def test_euclidean_distance_axis_out_of_bounds_raises_error():
    # Arrange
    from scitex_linalg import euclidean_distance

    uu = np.random.rand(3, 4)
    vv = np.random.rand(3, 4)
    # Act
    ctx = pytest.raises((IndexError, ValueError, Exception))
    # Assert
    with ctx:
        euclidean_distance(uu, vv, axis=5)


# ---------------------------------------------------------------------------
# euclidean_distance — shape compatibility
# ---------------------------------------------------------------------------


def test_euclidean_distance_mismatched_axis_size_raises_value_error():
    # Arrange
    from scitex_linalg import euclidean_distance

    uu = np.random.rand(3, 4)
    vv = np.random.rand(5, 4)
    # Act
    ctx = pytest.raises(ValueError, match="Shape along axis")
    # Assert
    with ctx:
        euclidean_distance(uu, vv, axis=0)


def test_euclidean_distance_compatible_shapes_return_expected_shape():
    # Arrange
    from scitex_linalg import euclidean_distance

    uu = np.random.rand(3, 4, 5)
    vv = np.random.rand(3, 2, 7)
    # Act
    dist = euclidean_distance(uu, vv, axis=0)
    # Assert
    assert dist.shape == (4, 5, 2, 7)


def test_euclidean_distance_3x2_arrays_axis0_returns_2_by_2_shape():
    # Arrange
    from scitex_linalg import euclidean_distance

    uu = np.array([[1, 2], [3, 4], [5, 6]])
    vv = np.array([[7, 8], [9, 10], [11, 12]])
    # Act
    dist = euclidean_distance(uu, vv, axis=0)
    # Assert
    assert dist.shape == (2, 2)


def test_euclidean_distance_3x2_arrays_axis0_yields_non_negative_values():
    # Arrange
    from scitex_linalg import euclidean_distance

    uu = np.array([[1, 2], [3, 4], [5, 6]])
    vv = np.array([[7, 8], [9, 10], [11, 12]])
    # Act
    dist = euclidean_distance(uu, vv, axis=0)
    # Assert
    assert np.all(dist >= 0)


# ---------------------------------------------------------------------------
# euclidean_distance — numeric accuracy
# ---------------------------------------------------------------------------


def test_euclidean_distance_3_4_5_right_triangle_returns_5():
    # Arrange
    from scitex_linalg import euclidean_distance

    uu = np.array([0, 0])
    vv = np.array([3, 4])
    # Act
    dist = euclidean_distance(uu, vv, axis=0)
    # Assert
    assert np.allclose(dist, 5.0)


def test_euclidean_distance_orthogonal_unit_vectors_returns_sqrt2():
    # Arrange
    from scitex_linalg import euclidean_distance

    uu = np.array([1, 0, 0])
    vv = np.array([0, 1, 0])
    # Act
    dist = euclidean_distance(uu, vv, axis=0)
    # Assert
    assert np.allclose(dist, np.sqrt(2))


def test_euclidean_distance_large_values_remains_numerically_stable():
    # Arrange
    from scitex_linalg import euclidean_distance

    uu = np.array([1e10, 1e10])
    vv = np.array([1e10 + 1, 1e10 + 1])
    # Act
    dist = euclidean_distance(uu, vv, axis=0)
    # Assert
    assert np.isclose(dist, np.sqrt(2), atol=1e-5)


def test_euclidean_distance_small_values_returns_expected_value():
    # Arrange
    from scitex_linalg import euclidean_distance

    uu = np.array([1e-10, 1e-10])
    vv = np.array([2e-10, 2e-10])
    # Act
    dist = euclidean_distance(uu, vv, axis=0)
    # Assert
    assert np.isclose(dist, np.sqrt(2) * 1e-10)


def test_euclidean_distance_mixed_sign_values_returns_expected_distance():
    # Arrange
    from scitex_linalg import euclidean_distance

    uu = np.array([-1, -2, -3])
    vv = np.array([1, 2, 3])
    expected = np.sqrt(4 + 16 + 36)
    # Act
    dist = euclidean_distance(uu, vv, axis=0)
    # Assert
    assert np.isclose(dist, expected)


# ---------------------------------------------------------------------------
# cdist wrapper
# ---------------------------------------------------------------------------


def test_cdist_basic_returns_3_by_3_shape():
    # Arrange
    from scitex_linalg import cdist

    XA = np.array([[0, 0], [1, 1], [2, 2]])
    XB = np.array([[0, 1], [1, 0], [3, 3]])
    # Act
    distances = cdist(XA, XB)
    # Assert
    assert distances.shape == (3, 3)


def test_cdist_basic_first_element_equals_one():
    # Arrange
    from scitex_linalg import cdist

    XA = np.array([[0, 0], [1, 1], [2, 2]])
    XB = np.array([[0, 1], [1, 0], [3, 3]])
    # Act
    distances = cdist(XA, XB)
    # Assert
    assert np.isclose(distances[0, 0], 1.0)


def test_cdist_basic_last_element_equals_sqrt2():
    # Arrange
    from scitex_linalg import cdist

    XA = np.array([[0, 0], [1, 1], [2, 2]])
    XB = np.array([[0, 1], [1, 0], [3, 3]])
    # Act
    distances = cdist(XA, XB)
    # Assert
    assert np.isclose(distances[2, 2], np.sqrt(2))


@pytest.mark.parametrize(
    "metric", ["euclidean", "cityblock", "cosine", "correlation"]
)
def test_cdist_matches_scipy_for_supported_metric(metric):
    # Arrange
    from scitex_linalg import cdist

    np.random.seed(0)
    XA = np.random.rand(5, 3)
    XB = np.random.rand(4, 3)
    expected = scipy_distance.cdist(XA, XB, metric=metric)
    # Act
    dist = cdist(XA, XB, metric=metric)
    # Assert
    assert np.allclose(dist, expected)


def test_cdist_custom_metric_first_pair_matches_l1_norm():
    # Arrange
    from scitex_linalg import cdist

    def custom_metric(u, v):
        return np.sum(np.abs(u - v))

    XA = np.array([[1, 2], [3, 4]])
    XB = np.array([[5, 6], [7, 8]])
    # Act
    dist = cdist(XA, XB, metric=custom_metric)
    # Assert
    assert np.isclose(dist[0, 0], 8)


def test_cdist_custom_metric_diagonal_pair_matches_l1_norm():
    # Arrange
    from scitex_linalg import cdist

    def custom_metric(u, v):
        return np.sum(np.abs(u - v))

    XA = np.array([[1, 2], [3, 4]])
    XB = np.array([[5, 6], [7, 8]])
    # Act
    dist = cdist(XA, XB, metric=custom_metric)
    # Assert
    assert np.isclose(dist[1, 1], 8)


def test_cdist_minkowski_p_kwarg_changes_result():
    # Arrange
    from scitex_linalg import cdist

    np.random.seed(1)
    XA = np.random.rand(3, 5)
    XB = np.random.rand(4, 5)
    # Act
    dist_p1 = cdist(XA, XB, metric="minkowski", p=1)
    dist_p2 = cdist(XA, XB, metric="minkowski", p=2)
    # Assert
    assert not np.allclose(dist_p1, dist_p2)


# ---------------------------------------------------------------------------
# edist alias
# ---------------------------------------------------------------------------


def test_edist_is_identity_alias_for_euclidean_distance():
    # Arrange
    from scitex_linalg import edist, euclidean_distance

    # Act
    same_object = edist is euclidean_distance
    # Assert
    assert same_object


def test_edist_has_same_name_as_euclidean_distance():
    # Arrange
    from scitex_linalg import edist, euclidean_distance

    # Act
    name = edist.__name__
    # Assert
    assert name == euclidean_distance.__name__


def test_edist_has_same_docstring_as_euclidean_distance():
    # Arrange
    from scitex_linalg import edist, euclidean_distance

    # Act
    doc = edist.__doc__
    # Assert
    assert doc == euclidean_distance.__doc__


def test_edist_returns_identical_output_to_euclidean_distance():
    # Arrange
    from scitex_linalg import edist, euclidean_distance

    np.random.seed(0)
    uu = np.random.rand(5, 3)
    vv = np.random.rand(5, 3)
    expected = euclidean_distance(uu, vv, axis=0)
    # Act
    actual = edist(uu, vv, axis=0)
    # Assert
    assert np.array_equal(actual, expected)


# ---------------------------------------------------------------------------
# @numpy_fn decorator — input type handling
# ---------------------------------------------------------------------------


def test_euclidean_distance_accepts_torch_tensor_input():
    # Arrange
    from scitex_linalg import euclidean_distance

    uu = torch.tensor([1.0, 2.0, 3.0])
    vv = torch.tensor([4.0, 5.0, 6.0])
    expected = np.sqrt((4 - 1) ** 2 + (5 - 2) ** 2 + (6 - 3) ** 2)
    # Act
    dist = euclidean_distance(uu, vv, axis=0)
    # Assert
    assert np.isclose(dist, expected)


def test_euclidean_distance_torch_input_returns_numpy_type():
    # Arrange
    from scitex_linalg import euclidean_distance

    uu = torch.tensor([1.0, 2.0, 3.0])
    vv = torch.tensor([4.0, 5.0, 6.0])
    # Act
    dist = euclidean_distance(uu, vv, axis=0)
    # Assert
    assert isinstance(dist, (np.ndarray, np.floating))


def test_euclidean_distance_python_list_inputs_return_correct_value():
    # Arrange
    from scitex_linalg import euclidean_distance

    uu = [1, 2, 3]
    vv = [4, 5, 6]
    expected = np.sqrt(27)
    # Act
    dist = euclidean_distance(uu, vv, axis=0)
    # Assert
    assert np.isclose(dist, expected)


def test_euclidean_distance_list_input_returns_numpy_type():
    # Arrange
    from scitex_linalg import euclidean_distance

    uu = [1, 2, 3]
    vv = [4, 5, 6]
    # Act
    dist = euclidean_distance(uu, vv, axis=0)
    # Assert
    assert isinstance(dist, (np.ndarray, np.floating))


def test_euclidean_distance_accepts_mixed_array_and_list_inputs():
    # Arrange
    from scitex_linalg import euclidean_distance

    uu = np.array([1.0, 2.0, 3.0])
    vv = [4, 5, 6]
    # Act
    dist = euclidean_distance(uu, vv, axis=0)
    # Assert
    assert isinstance(dist, (np.ndarray, np.floating))


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_euclidean_distance_empty_arrays_return_zero():
    # Arrange
    from scitex_linalg import euclidean_distance

    uu = np.array([])
    vv = np.array([])
    # Act
    dist = euclidean_distance(uu, vv, axis=0)
    # Assert
    assert dist == 0.0


def test_euclidean_distance_nan_input_propagates_nan_output():
    # Arrange
    from scitex_linalg import euclidean_distance

    uu = np.array([1, 2, np.nan])
    vv = np.array([4, 5, 6])
    # Act
    dist = euclidean_distance(uu, vv, axis=0)
    # Assert
    assert np.isnan(dist)


def test_euclidean_distance_inf_input_propagates_inf_output():
    # Arrange
    from scitex_linalg import euclidean_distance

    uu = np.array([1, 2, np.inf])
    vv = np.array([4, 5, 6])
    # Act
    dist = euclidean_distance(uu, vv, axis=0)
    # Assert
    assert np.isinf(dist)


# ---------------------------------------------------------------------------
# Performance smoke tests
# ---------------------------------------------------------------------------


def test_euclidean_distance_large_arrays_complete_under_one_second():
    # Arrange
    import time

    from scitex_linalg import euclidean_distance

    uu = np.random.rand(100, 50)
    vv = np.random.rand(100, 50)
    # Act
    start = time.time()
    euclidean_distance(uu, vv, axis=0)
    duration = time.time() - start
    # Assert
    assert duration < 1.0


def test_euclidean_distance_large_arrays_return_correct_shape():
    # Arrange
    from scitex_linalg import euclidean_distance

    uu = np.random.rand(100, 50)
    vv = np.random.rand(100, 50)
    # Act
    dist = euclidean_distance(uu, vv, axis=0)
    # Assert
    assert dist.shape == (50, 50)


def test_euclidean_distance_memory_efficient_for_1000_by_10_inputs():
    # Arrange
    from scitex_linalg import euclidean_distance

    uu = np.random.rand(1000, 10)
    vv = np.random.rand(1000, 10)
    # Act
    dist = euclidean_distance(uu, vv, axis=0)
    # Assert
    assert dist.shape == (10, 10)


# ---------------------------------------------------------------------------
# Documentation
# ---------------------------------------------------------------------------


def test_euclidean_distance_docstring_is_present():
    # Arrange
    from scitex_linalg import euclidean_distance

    # Act
    doc = euclidean_distance.__doc__
    # Assert
    assert doc is not None


def test_euclidean_distance_docstring_describes_euclidean_distance():
    # Arrange
    from scitex_linalg import euclidean_distance

    # Act
    doc = euclidean_distance.__doc__
    # Assert
    assert "Euclidean distance" in doc


def test_euclidean_distance_docstring_documents_parameters_section():
    # Arrange
    from scitex_linalg import euclidean_distance

    # Act
    doc = euclidean_distance.__doc__
    # Assert
    assert "Parameters" in doc


def test_euclidean_distance_docstring_documents_returns_section():
    # Arrange
    from scitex_linalg import euclidean_distance

    # Act
    doc = euclidean_distance.__doc__
    # Assert
    assert "Returns" in doc


def test_cdist_docstring_matches_scipy_cdist_docstring():
    # Arrange
    from scitex_linalg import cdist

    # Act
    doc = cdist.__doc__
    # Assert
    assert doc == scipy_distance.cdist.__doc__


# ---------------------------------------------------------------------------
# Cross-implementation comparison
# ---------------------------------------------------------------------------


def test_euclidean_distance_matches_scipy_euclidean_for_simple_vectors():
    # Arrange
    from scipy.spatial.distance import euclidean

    from scitex_linalg import euclidean_distance

    u = np.array([1, 2, 3])
    v = np.array([4, 5, 6])
    expected = euclidean(u, v)
    # Act
    actual = euclidean_distance(u, v, axis=0)
    # Assert
    assert np.isclose(actual, expected)


def test_cdist_pairwise_unit_square_returns_unit_horizontal_distance():
    # Arrange
    from scitex_linalg import cdist

    points = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    # Act
    dist_cdist = cdist(points, points)
    # Assert
    assert np.isclose(dist_cdist[0, 1], 1.0)


def test_cdist_pairwise_unit_square_returns_sqrt2_diagonal_distance():
    # Arrange
    from scitex_linalg import cdist

    points = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    # Act
    dist_cdist = cdist(points, points)
    # Assert
    assert np.isclose(dist_cdist[0, 3], np.sqrt(2))


if __name__ == "__main__":
    import os

    pytest.main([os.path.abspath(__file__)])
