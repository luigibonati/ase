import pytest
import numpy as np
from ase.dft.wannier import gram_schmidt

@pytest.fixture
def rng():
    return np.random.RandomState(0)


def orthonormality_error(matrix):
    return np.abs(matrix.T @ matrix - np.eye(len(matrix))).max()


def test_gram_schmidt(rng):
    matrix = rng.rand(4, 4)
    assert orthonormality_error(matrix) > 1
    gram_schmidt(matrix)
    assert orthonormality_error(matrix) < 1e-12
