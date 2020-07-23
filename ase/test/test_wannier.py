import pytest
import numpy as np
from ase.transport.tools import dagger
from ase.dft.kpoints import monkhorst_pack
from ase.dft.wannier import gram_schmidt, lowdin, random_orthogonal_matrix, \
    neighbor_k_search, calculate_weights, steepest_descent, md_min


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


def test_gram_schmidt(rng):
    matrix = rng.rand(4, 4)
    assert orthonormality_error(matrix) > 1
    gram_schmidt(matrix)
    assert orthonormality_error(matrix) < 1e-12


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


def test_calculate_weights():
    # Equation from Berghold et al. PRB v61 n15 (2000)
    tol = 1e-5
    cubic = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    g_sc = np.dot(cubic, cubic.T)
    w_sc, G_sc = calculate_weights(cubic, normalize=False)
    bcc = np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]], dtype=float)
    g_bcc = np.dot(bcc, bcc.T)
    w_bcc, G_bcc = calculate_weights(bcc, normalize=False)
    fcc = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float)
    g_fcc = np.dot(fcc, fcc.T)
    w_fcc, G_fcc = calculate_weights(fcc, normalize=False)

    errors_sc = []
    errors_bcc = []
    errors_fcc = []
    for i in range(3):
        for j in range(3):
            errors_sc.append(np.abs(np.dot(w_sc * G_sc[:, i], G_sc[:, j])
                                    - g_sc[i, j]))
            errors_bcc.append(np.abs(np.dot(w_bcc * G_bcc[:, i], G_bcc[:, j])
                                     - g_bcc[i, j]))
            errors_fcc.append(np.abs(np.dot(w_fcc * G_fcc[:, i], G_fcc[:, j])
                                     - g_fcc[i, j]))

    assert np.max(errors_sc) < tol, 'Wrong weights for simple cubic cell'
    assert np.max(errors_bcc) < tol, 'Wrong weights for bcc cell'
    assert np.max(errors_fcc) < tol, 'Wrong weights for fcc cell'


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
