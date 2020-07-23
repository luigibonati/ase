import pytest
import numpy as np
from ase.transport.tools import dagger
from ase.dft.kpoints import monkhorst_pack
from ase.dft.wannier import gram_schmidt, lowdin, random_orthogonal_matrix, \
    neighbor_k_search, steepest_descent, md_min


@pytest.fixture
def rng():
    return np.random.RandomState(0)


class paraboloid:

    def __init__(self, pos=np.array([10, 10, 10], dtype=complex)):
        self.pos = pos

    def get_gradients(self):
        return 2 * self.pos

    def step(self, dF, updaterot=True, updatecoeff=True):
        self.pos -= dF

    def get_functional_value(self):
        return np.sum(self.pos**2)


def orthonormality_error(matrix):
    return np.abs(dagger(matrix) @ matrix - np.eye(len(matrix))).max()


# def orthogonality_error_single(matrix, column):
#     # test orthogonality of the columns of the matrix to a single column
#     indices = list(range(len(matrix.T)))
#     del indices[indices.index(column)]
#     errors = [np.abs(matrix.T[i] @ matrix[:, column]) for i in indices]
#     return np.max(errors)


def test_gram_schmidt(rng):
    matrix = rng.rand(4, 4)
    assert orthonormality_error(matrix) > 1
    gram_schmidt(matrix)
    assert orthonormality_error(matrix) < 1e-12


# def test_gram_schmidt_single(rng):
#     N = 4
#     matrix = rng.rand(N, N)
#     assert orthonormality_error(matrix) > 1
#     for i in range(N):
#         gram_schmidt_single(matrix, i)
#         assert orthogonality_error_single(matrix, i) < 1e-12


def test_lowdin(rng):
    matrix = rng.rand(4, 4)
    assert orthonormality_error(matrix) > 1
    lowdin(matrix)
    assert orthonormality_error(matrix) < 1e-12


def test_random_orthogonal_matrix(rng):
    dim = 4
    matrix = random_orthogonal_matrix(dim, rng=rng, real=True)
    assert matrix.shape[0] == matrix.shape[1]
    assert orthonormality_error(matrix) < 1e-12
    matrix = random_orthogonal_matrix(dim, rng=rng, real=False)
    assert matrix.shape[0] == matrix.shape[1]
    assert orthonormality_error(matrix) < 1e-12


def test_neighbor_k_search():
    kpt_kc = monkhorst_pack((4, 4, 4))
    Gdir_dc = [[1, 0, 0], [0, 1, 0], [0, 0, 1],
               [1, 1, 0], [1, 0, 1], [0, 1, 1]]
    tol = 1e-4
    for d, Gdir_c in enumerate(Gdir_dc):
        for k, k_c in enumerate(kpt_kc):
            kk, k0 = neighbor_k_search(k_c, Gdir_c, kpt_kc, tol=tol)
            assert np.linalg.norm(kpt_kc[kk] - k_c - Gdir_c + k0) < tol


def test_steepest_descent():
    tol = 0.1
    step = 0.1
    func = paraboloid(pos=np.array([10, 10, 10], dtype=float))
    steepest_descent(func=func, step=step, tolerance=tol, verbose=False)
    assert func.get_functional_value() < 0.1


def test_md_min():
    tol = 1e-3
    step = 0.1
    func = paraboloid(pos=np.array([10, 10, 10], dtype=complex))
    md_min(func=func, step=step, tolerance=tol, verbose=False)
    assert func.get_functional_value() < 0.1
