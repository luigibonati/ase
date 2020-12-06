import pytest


def compare_single_nested_arrays(quantity, expected_values, rel_tol):
    for vec, vec_expected in zip(quantity, expected_values):
        assert vec == pytest.approx(vec_expected, rel=rel_tol)
