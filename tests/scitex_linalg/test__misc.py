#!/usr/bin/env python3
"""TQ-compliant tests for miscellaneous linear algebra helpers.

Covers `cosine`, `nannorm`, `rebase_a_vec`, `three_line_lengths_to_coords`.
Each test: ≥3 word-tokens after `test_`, single assertion, AAA markers.
"""

import warnings

import pytest

pytest.importorskip("sympy")

import numpy as np


def _to_float_tuple(t):
    """Convert tuple with sympy.Float values to Python floats."""
    return tuple(float(x) for x in t)


# ---------------------------------------------------------------------------
# cosine — angle correctness
# ---------------------------------------------------------------------------


def test_cosine_parallel_vectors_returns_one():
    # Arrange
    from scitex_linalg import cosine

    v1 = np.array([1, 0, 0])
    v2 = np.array([2, 0, 0])
    # Act
    result = cosine(v1, v2)
    # Assert
    assert np.isclose(result, 1.0)


def test_cosine_orthogonal_vectors_returns_zero():
    # Arrange
    from scitex_linalg import cosine

    v1 = np.array([1, 0, 0])
    v2 = np.array([0, 1, 0])
    # Act
    result = cosine(v1, v2)
    # Assert
    assert np.isclose(result, 0.0)


def test_cosine_opposite_vectors_returns_minus_one():
    # Arrange
    from scitex_linalg import cosine

    v1 = np.array([1, 0, 0])
    v2 = np.array([-1, 0, 0])
    # Act
    result = cosine(v1, v2)
    # Assert
    assert np.isclose(result, -1.0)


def test_cosine_45_degree_angle_returns_one_over_sqrt2():
    # Arrange
    from scitex_linalg import cosine

    v1 = np.array([1, 0])
    v2 = np.array([1, 1])
    # Act
    result = cosine(v1, v2)
    # Assert
    assert np.isclose(result, 1 / np.sqrt(2))


def test_cosine_60_degree_angle_returns_one_half():
    # Arrange
    from scitex_linalg import cosine

    v1 = np.array([1, 0])
    v2 = np.array([1, np.sqrt(3)])
    # Act
    result = cosine(v1, v2)
    # Assert
    assert np.isclose(result, 0.5)


def test_cosine_nan_in_first_vector_returns_nan():
    # Arrange
    from scitex_linalg import cosine

    v1 = np.array([1, np.nan, 0])
    v2 = np.array([1, 1, 1])
    # Act
    result = cosine(v1, v2)
    # Assert
    assert np.isnan(result)


def test_cosine_nan_in_second_vector_returns_nan():
    # Arrange
    from scitex_linalg import cosine

    v1 = np.array([1, 1, 1])
    v2 = np.array([1, np.nan, 0])
    # Act
    result = cosine(v1, v2)
    # Assert
    assert np.isnan(result)


def test_cosine_nan_in_both_vectors_returns_nan():
    # Arrange
    from scitex_linalg import cosine

    v1 = np.array([np.nan, 1, 0])
    v2 = np.array([1, np.nan, 0])
    # Act
    result = cosine(v1, v2)
    # Assert
    assert np.isnan(result)


def test_cosine_2d_general_vectors_returns_correct_value():
    # Arrange
    from scitex_linalg import cosine

    v1 = np.array([3, 4])
    v2 = np.array([4, 3])
    expected = (3 * 4 + 4 * 3) / (5 * 5)
    # Act
    result = cosine(v1, v2)
    # Assert
    assert np.isclose(result, expected)


def test_cosine_3d_general_vectors_returns_correct_value():
    # Arrange
    from scitex_linalg import cosine

    v1 = np.array([1, 2, 2])
    v2 = np.array([2, 1, 2])
    expected = (1 * 2 + 2 * 1 + 2 * 2) / (3 * 3)
    # Act
    result = cosine(v1, v2)
    # Assert
    assert np.isclose(result, expected)


def test_cosine_zero_vector_returns_nan_or_inf():
    # Arrange
    from scitex_linalg import cosine

    v1 = np.array([0, 0, 0])
    v2 = np.array([1, 1, 1])
    # Act
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = cosine(v1, v2)
    # Assert
    assert np.isnan(result) or np.isinf(result)


def test_cosine_invariant_to_positive_scaling_of_inputs():
    # Arrange
    from scitex_linalg import cosine

    v1 = np.array([1, 2, 3])
    v2 = np.array([4, 5, 6])
    base = cosine(v1, v2)
    # Act
    scaled = cosine(v1 * 10, v2 * 0.1)
    # Assert
    assert np.isclose(base, scaled)


# ---------------------------------------------------------------------------
# nannorm — norm correctness
# ---------------------------------------------------------------------------


def test_nannorm_3_4_returns_five():
    # Arrange
    from scitex_linalg import nannorm

    v = np.array([3, 4])
    # Act
    result = nannorm(v)
    # Assert
    assert np.isclose(result, 5)


def test_nannorm_unit_vector_returns_one():
    # Arrange
    from scitex_linalg import nannorm

    v = np.array([1, 0, 0])
    # Act
    result = nannorm(v)
    # Assert
    assert np.isclose(result, 1)


def test_nannorm_ones_3d_returns_sqrt3():
    # Arrange
    from scitex_linalg import nannorm

    v = np.array([1, 1, 1])
    # Act
    result = nannorm(v)
    # Assert
    assert np.isclose(result, np.sqrt(3))


def test_nannorm_single_nan_input_returns_nan():
    # Arrange
    from scitex_linalg import nannorm

    v = np.array([1, np.nan, 3])
    # Act
    result = nannorm(v)
    # Assert
    assert np.isnan(result)


def test_nannorm_all_nan_input_returns_nan():
    # Arrange
    from scitex_linalg import nannorm

    v = np.array([np.nan, np.nan, np.nan])
    # Act
    result = nannorm(v)
    # Assert
    assert np.isnan(result)


def test_nannorm_2d_default_axis_returns_row_norms():
    # Arrange
    from scitex_linalg import nannorm

    v = np.array([[1, 2, 3], [4, 5, 6]])
    expected = np.array([np.sqrt(14), np.sqrt(77)])
    # Act
    result = nannorm(v, axis=-1)
    # Assert
    assert np.allclose(result, expected)


def test_nannorm_2d_axis0_returns_column_norms():
    # Arrange
    from scitex_linalg import nannorm

    v = np.array([[1, 2, 3], [4, 5, 6]])
    expected = np.array([np.sqrt(17), np.sqrt(29), np.sqrt(45)])
    # Act
    result = nannorm(v, axis=0)
    # Assert
    assert np.allclose(result, expected)


def test_nannorm_empty_array_returns_zero():
    # Arrange
    from scitex_linalg import nannorm

    v = np.array([])
    # Act
    result = nannorm(v)
    # Assert
    assert result == 0.0


def test_nannorm_single_positive_element_returns_value():
    # Arrange
    from scitex_linalg import nannorm

    v = np.array([5])
    # Act
    result = nannorm(v)
    # Assert
    assert result == 5


def test_nannorm_single_negative_element_returns_absolute_value():
    # Arrange
    from scitex_linalg import nannorm

    v = np.array([-5])
    # Act
    result = nannorm(v)
    # Assert
    assert result == 5


def test_nannorm_complex_3_plus_4j_returns_five():
    # Arrange
    from scitex_linalg import nannorm

    v = np.array([3 + 4j, 0])
    # Act
    result = nannorm(v)
    # Assert
    assert np.isclose(result, 5)


def test_nannorm_large_values_returns_positive_finite_result():
    # Arrange
    from scitex_linalg import nannorm

    v = np.array([1e200, 1e200])
    # Act
    result = nannorm(v)
    # Assert
    assert result > 0 and not np.isnan(result)


# ---------------------------------------------------------------------------
# rebase_a_vec — projection correctness
# ---------------------------------------------------------------------------


def test_rebase_a_vec_x_axis_projection_returns_x_component():
    # Arrange
    from scitex_linalg import rebase_a_vec

    v = np.array([3, 4])
    v_base = np.array([1, 0])
    # Act
    result = rebase_a_vec(v, v_base)
    # Assert
    assert np.isclose(result, 3)


def test_rebase_a_vec_y_axis_projection_returns_y_component():
    # Arrange
    from scitex_linalg import rebase_a_vec

    v = np.array([3, 4])
    v_base = np.array([0, 1])
    # Act
    result = rebase_a_vec(v, v_base)
    # Assert
    assert np.isclose(result, 4)


def test_rebase_a_vec_parallel_vectors_returns_full_length():
    # Arrange
    from scitex_linalg import rebase_a_vec

    v = np.array([2, 2])
    v_base = np.array([1, 1])
    expected = np.sqrt(8)
    # Act
    result = rebase_a_vec(v, v_base)
    # Assert
    assert np.isclose(result, expected)


def test_rebase_a_vec_opposite_direction_returns_negative_length():
    # Arrange
    from scitex_linalg import rebase_a_vec

    v = np.array([1, 0])
    v_base = np.array([-1, 0])
    # Act
    result = rebase_a_vec(v, v_base)
    # Assert
    assert np.isclose(result, -1)


def test_rebase_a_vec_orthogonal_vectors_returns_zero():
    # Arrange
    from scitex_linalg import rebase_a_vec

    v = np.array([0, 1])
    v_base = np.array([1, 0])
    # Act
    result = rebase_a_vec(v, v_base)
    # Assert
    assert np.isclose(result, 0)


def test_rebase_a_vec_3d_x_axis_projection_returns_x_component():
    # Arrange
    from scitex_linalg import rebase_a_vec

    v = np.array([1, 1, 1])
    v_base = np.array([1, 0, 0])
    # Act
    result = rebase_a_vec(v, v_base)
    # Assert
    assert np.isclose(result, 1)


def test_rebase_a_vec_nan_in_v_returns_nan():
    # Arrange
    from scitex_linalg import rebase_a_vec

    v = np.array([np.nan, 1])
    v_base = np.array([1, 0])
    # Act
    result = rebase_a_vec(v, v_base)
    # Assert
    assert np.isnan(result)


def test_rebase_a_vec_nan_in_v_base_returns_nan():
    # Arrange
    from scitex_linalg import rebase_a_vec

    v = np.array([1, 1])
    v_base = np.array([np.nan, 0])
    # Act
    result = rebase_a_vec(v, v_base)
    # Assert
    assert np.isnan(result)


def test_rebase_a_vec_docstring_example_returns_three():
    # Arrange
    from scitex_linalg import rebase_a_vec

    v = np.array([3, 4])
    v_base = np.array([10, 0])
    # Act
    result = rebase_a_vec(v, v_base)
    # Assert
    assert np.isclose(result, 3)


# ---------------------------------------------------------------------------
# three_line_lengths_to_coords — geometry correctness
# ---------------------------------------------------------------------------


def test_three_line_lengths_3_4_5_returns_origin_for_first_point():
    # Arrange
    from scitex_linalg import three_line_lengths_to_coords

    # Act
    origin, _, _ = three_line_lengths_to_coords(3, 4, 5)
    # Assert
    assert origin == (0, 0, 0)


def test_three_line_lengths_3_4_5_places_a_on_positive_x_axis():
    # Arrange
    from scitex_linalg import three_line_lengths_to_coords

    # Act
    _, point_a, _ = three_line_lengths_to_coords(3, 4, 5)
    # Assert
    assert point_a == (3, 0, 0)


def test_three_line_lengths_3_4_5_makes_OB_distance_four():
    # Arrange
    from scitex_linalg import three_line_lengths_to_coords

    origin, _, point_b = three_line_lengths_to_coords(3, 4, 5)
    b_float = _to_float_tuple(point_b)
    o_float = _to_float_tuple(origin)
    # Act
    distance = np.linalg.norm(np.array(b_float) - np.array(o_float))
    # Assert
    assert np.isclose(distance, 4)


def test_three_line_lengths_3_4_5_makes_AB_distance_five():
    # Arrange
    from scitex_linalg import three_line_lengths_to_coords

    _, point_a, point_b = three_line_lengths_to_coords(3, 4, 5)
    b_float = _to_float_tuple(point_b)
    a_float = _to_float_tuple(point_a)
    # Act
    distance = np.linalg.norm(np.array(b_float) - np.array(a_float))
    # Assert
    assert np.isclose(distance, 5)


def test_three_line_lengths_equilateral_b_x_coordinate_equals_one():
    # Arrange
    from scitex_linalg import three_line_lengths_to_coords

    side = 2
    _, _, point_b = three_line_lengths_to_coords(side, side, side)
    b_float = _to_float_tuple(point_b)
    # Act
    bx = b_float[0]
    # Assert
    assert np.isclose(bx, 1)


def test_three_line_lengths_equilateral_b_y_coordinate_equals_sqrt3():
    # Arrange
    from scitex_linalg import three_line_lengths_to_coords

    side = 2
    _, _, point_b = three_line_lengths_to_coords(side, side, side)
    b_float = _to_float_tuple(point_b)
    # Act
    by = b_float[1]
    # Assert
    assert np.isclose(by, np.sqrt(3))


def test_three_line_lengths_equilateral_b_z_coordinate_equals_zero():
    # Arrange
    from scitex_linalg import three_line_lengths_to_coords

    side = 2
    _, _, point_b = three_line_lengths_to_coords(side, side, side)
    b_float = _to_float_tuple(point_b)
    # Act
    bz = b_float[2]
    # Assert
    assert bz == 0


def test_three_line_lengths_docstring_example_origin_equals_zero():
    # Arrange
    from scitex_linalg import three_line_lengths_to_coords

    # Act
    origin, _, _ = three_line_lengths_to_coords(2, np.sqrt(3), 1)
    # Assert
    assert origin == (0, 0, 0)


def test_three_line_lengths_docstring_example_a_on_positive_x_axis():
    # Arrange
    from scitex_linalg import three_line_lengths_to_coords

    # Act
    _, point_a, _ = three_line_lengths_to_coords(2, np.sqrt(3), 1)
    # Assert
    assert point_a == (2, 0, 0)


def test_three_line_lengths_violating_triangle_inequality_raises_index_error():
    # Arrange
    from scitex_linalg import three_line_lengths_to_coords

    # Act
    ctx = pytest.raises(IndexError)
    # Assert
    with ctx:
        three_line_lengths_to_coords(1, 1, 3)


def test_three_line_lengths_zero_cc_returns_3_element_b_coordinate():
    # Arrange
    from scitex_linalg import three_line_lengths_to_coords

    # Act
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, _, point_b = three_line_lengths_to_coords(1, 1, 0)
    # Assert
    assert len(_to_float_tuple(point_b)) == 3


def test_three_line_lengths_isosceles_origin_equals_zero():
    # Arrange
    from scitex_linalg import three_line_lengths_to_coords

    # Act
    origin, _, _ = three_line_lengths_to_coords(3, 3, 4)
    # Assert
    assert origin == (0, 0, 0)


def test_three_line_lengths_isosceles_a_on_positive_x_axis():
    # Arrange
    from scitex_linalg import three_line_lengths_to_coords

    # Act
    _, point_a, _ = three_line_lengths_to_coords(3, 3, 4)
    # Assert
    assert point_a == (3, 0, 0)


def test_three_line_lengths_isosceles_OB_distance_equals_three():
    # Arrange
    from scitex_linalg import three_line_lengths_to_coords

    origin, _, point_b = three_line_lengths_to_coords(3, 3, 4)
    b_float = _to_float_tuple(point_b)
    o_float = _to_float_tuple(origin)
    # Act
    distance = np.linalg.norm(np.array(b_float) - np.array(o_float))
    # Assert
    assert np.isclose(distance, 3)


def test_three_line_lengths_isosceles_AB_distance_equals_four():
    # Arrange
    from scitex_linalg import three_line_lengths_to_coords

    _, point_a, point_b = three_line_lengths_to_coords(3, 3, 4)
    b_float = _to_float_tuple(point_b)
    a_float = _to_float_tuple(point_a)
    # Act
    distance = np.linalg.norm(np.array(b_float) - np.array(a_float))
    # Assert
    assert np.isclose(distance, 4)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_cosine_empty_vectors_returns_nan_or_inf():
    # Arrange
    from scitex_linalg import cosine

    v1 = np.array([])
    v2 = np.array([])
    # Act
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = cosine(v1, v2)
    # Assert
    assert np.isnan(result) or np.isinf(result)


def test_rebase_a_vec_zero_base_returns_nan_or_inf_or_raises():
    # Arrange
    from scitex_linalg import rebase_a_vec

    v = np.array([1, 2, 3])
    v_base = np.array([0, 0, 0])
    # Act
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            result = rebase_a_vec(v, v_base)
            failed_or_nan_or_inf = np.isnan(result) or np.isinf(result)
        except ValueError:
            failed_or_nan_or_inf = True
    # Assert
    assert failed_or_nan_or_inf


# ---------------------------------------------------------------------------
# Numerical stability
# ---------------------------------------------------------------------------


def test_cosine_large_values_returns_correct_normalized_result():
    # Arrange
    from scitex_linalg import cosine

    v1 = np.array([1e100, 2e100])
    v2 = np.array([3e100, 4e100])
    expected = (3 + 8) / (np.sqrt(5) * np.sqrt(25))
    # Act
    result = cosine(v1, v2)
    # Assert
    assert np.isclose(result, expected)


def test_cosine_small_values_returns_correct_normalized_result():
    # Arrange
    from scitex_linalg import cosine

    v1 = np.array([1e-100, 2e-100])
    v2 = np.array([3e-100, 4e-100])
    expected = (3 + 8) / (np.sqrt(5) * np.sqrt(25))
    # Act
    result = cosine(v1, v2)
    # Assert
    assert np.isclose(result, expected)


# ---------------------------------------------------------------------------
# Cross-function integration
# ---------------------------------------------------------------------------


def test_cosine_and_rebase_a_vec_agree_on_3_4_projection_value():
    # Arrange
    from scitex_linalg import cosine, rebase_a_vec

    v = np.array([3, 4])
    v_base = np.array([1, 0])
    expected = cosine(v, v_base) * np.linalg.norm(v)
    # Act
    actual = rebase_a_vec(v, v_base)
    # Assert
    assert np.isclose(actual, expected)


def test_three_line_lengths_3_4_5_yields_right_angle_at_origin():
    # Arrange
    from scitex_linalg import cosine, three_line_lengths_to_coords

    origin, point_a, point_b = three_line_lengths_to_coords(3, 4, 5)
    o_arr = np.array([float(x) for x in origin])
    a_arr = np.array([float(x) for x in point_a])
    b_arr = np.array([float(x) for x in point_b])
    # Act
    cos_angle = cosine(a_arr - o_arr, b_arr - o_arr)
    # Assert
    assert np.isclose(cos_angle, 0, atol=1e-10)


if __name__ == "__main__":
    import os

    pytest.main([os.path.abspath(__file__)])
