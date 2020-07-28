import pytest
import numpy as np
from ase.transport.tools import dagger, normalize
from ase.dft.kpoints import monkhorst_pack
from ase.build import molecule
from ase.lattice import CUB, FCC, BCC, TET, BCT, ORC, ORCF, ORCI, ORCC, HEX, \
    RHL, MCL, MCLC, TRI, OBL, HEX2D, RECT, CRECT, SQR, LINE
from ase.dft.wannier import gram_schmidt, lowdin, random_orthogonal_matrix, \
    neighbor_k_search, calculate_weights, steepest_descent, md_min, \
    rotation_from_projection, Wannier

try:
    from gpaw import GPAW
except ImportError:
    raise unittest.SkipTest('GPAW not available')



@pytest.fixture
def rng():
    return np.random.RandomState(0)


class Paraboloid:

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


def orthogonality_error(matrix):
    errors = []
    for i in range(len(matrix)):
        for j in range(i + 1, len(matrix)):
            errors.append(np.abs(matrix[i].T @ matrix[j]))
    return np.max(errors)


def normalization_error(matrix):
    old_matrix = matrix.copy()
    normalize(matrix)
    return np.abs(matrix - old_matrix).max()


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


@pytest.mark.parametrize('lat', [CUB(1), FCC(1), BCC(1), TET(1, 2), BCT(1, 2),
                                 ORC(1, 2, 3), ORCF(1, 2, 3), ORCI(1, 2, 3),
                                 ORCC(1, 2, 3), HEX(1, 2), RHL(1, 110),
                                 MCL(1, 2, 3, 70), MCLC(1, 2, 3, 70),
                                 TRI(1, 2, 3, 60, 70, 80), OBL(1, 2, 110),
                                 HEX2D(1), RECT(1, 2), CRECT(1, 70), SQR(1),
                                 LINE(1)])
def test_calculate_weights(lat):
    # Equation from Berghold et al. PRB v61 n15 (2000)
    tol = 1e-5
    cell = lat.tocell()
    g = cell @ cell.T
    w, G = calculate_weights(cell, normalize=False)

    errors = []
    for i in range(3):
        for j in range(3):
            errors.append(np.abs((w * G[:, i] @ G[:, j]) - g[i, j]))

    assert np.max(errors) < tol


def test_steepest_descent():
    tol = 0.1
    step = 0.1
    func = Paraboloid(pos=np.array([10, 10, 10], dtype=float))
    steepest_descent(func=func, step=step, tolerance=tol, verbose=False)
    assert func.get_functional_value() < 0.1


def test_md_min():
    tol = 1e-3
    step = 0.1
    func = Paraboloid(pos=np.array([10, 10, 10], dtype=complex))
    md_min(func=func, step=step, tolerance=tol, verbose=False)
    assert func.get_functional_value() < 0.1


def test_rotation_from_projection(rng):
    proj_nw = rng.rand(6, 4)
    assert orthonormality_error(proj_nw[:int(min(proj_nw.shape))]) > 1
    U_ww, C_ul = rotation_from_projection(proj_nw, fixed=2, ortho=True)
    assert orthonormality_error(U_ww) < 1e-10, 'U_ww not unitary'
    assert orthogonality_error(C_ul.T) < 1e-10, 'C_ul columns not orthogonal'
    assert normalization_error(C_ul) < 1e-10, 'C_ul not normalized'
    U_ww, C_ul = rotation_from_projection(proj_nw, fixed=2, ortho=False)
    assert normalization_error(U_ww) < 1e-10, 'U_ww not normalized'


def test_save(tmpdir):
    calc = GPAW(gpts=(8, 8, 8), nbands=4, txt=None)
    atoms = molecule('H2', calculator=calc)
    atoms.center(vacuum=3.)
    atoms.get_potential_energy()
    wan1 = Wannier(nwannier=4, fixedstates=2, calc=calc, initialwannier='bloch')

    picklefile = tmpdir.join('wan.pickle')
    wan1.save(picklefile)

    wan2 = Wannier(nwannier=4, fixedstates=2, file=picklefile, calc=calc)

    assert np.abs(wan1.get_functional_value() -
                  wan2.get_functional_value()).max() < 1e-12


# The following test always fails because get_radii() is broken.
@pytest.mark.parametrize('lat', [CUB(1), FCC(1), BCC(1), TET(1, 2), BCT(1, 2),
                                 ORC(1, 2, 3), ORCF(1, 2, 3), ORCI(1, 2, 3),
                                 ORCC(1, 2, 3), HEX(1, 2), RHL(1, 110),
                                 MCL(1, 2, 3, 70), MCLC(1, 2, 3, 70),
                                 TRI(1, 2, 3, 60, 70, 80), OBL(1, 2, 110),
                                 HEX2D(1), RECT(1, 2), CRECT(1, 70), SQR(1),
                                 LINE(1)])
def test_get_radii(lat):
    calc = GPAW(gpts=(8, 8, 8), nbands=4, txt=None)
    atoms = molecule('H2', calculator=calc, pbc=True)
    atoms.cell = lat.tocell()
    atoms.center(vacuum=3.)
    atoms.get_potential_energy()
    wan = Wannier(nwannier=4, fixedstates=2, calc=calc, initialwannier='bloch')
    assert not (wan.get_radii() == 0).all()
