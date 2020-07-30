import pytest
import numpy as np
from ase.transport.tools import dagger, normalize
from ase.dft.kpoints import monkhorst_pack
from ase.build import molecule
from ase.io.cube import read_cube
from ase.lattice import CUB, FCC, BCC, TET, BCT, ORC, ORCF, ORCI, ORCC, HEX, \
    RHL, MCL, MCLC, TRI, OBL, HEX2D, RECT, CRECT, SQR, LINE
from ase.dft.wannier import gram_schmidt, lowdin, random_orthogonal_matrix, \
    neighbor_k_search, calculate_weights, steepest_descent, md_min, \
    rotation_from_projection, Wannier


@pytest.fixture
def rng():
    return np.random.RandomState(0)


@pytest.fixture
def wan(rng):
    def _wan(gpts=(8, 8, 8),
             atoms=None,
             calc=None,
             nwannier=2,
             fixedstates=None,
             initialwannier='bloch',
             kpts=(1, 1, 1),
             file=None,
             rng=rng):
        if calc is None:
            gpaw = pytest.importorskip('gpaw')
            calc = gpaw.GPAW(gpts=gpts, nbands=nwannier, kpts=kpts,
                             symmetry='off', txt=None)
        if atoms is None:
            pbc = (np.array(kpts) > 1).any()
            atoms = molecule('H2', pbc=pbc)
            atoms.center(vacuum=3.)
        atoms.calc = calc
        atoms.get_potential_energy()
        return Wannier(nwannier=nwannier,
                       fixedstates=fixedstates,
                       calc=calc,
                       initialwannier=initialwannier,
                       file=None,
                       rng=rng)
    return _wan


class Paraboloid:

    def __init__(self, pos=np.array([10., 10., 10.], dtype=complex), shift=1.):
        self.pos = pos
        self.shift = shift

    def get_gradients(self):
        return 2 * self.pos

    def step(self, dF, updaterot=True, updatecoeff=True):
        self.pos -= dF

    def get_functional_value(self):
        return np.sum(self.pos**2) + self.shift


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
    tol = 1e-6
    step = 0.1
    func = Paraboloid(pos=np.array([10, 10, 10], dtype=float), shift=1.)
    steepest_descent(func=func, step=step, tolerance=tol, verbose=False)
    assert func.get_functional_value() == pytest.approx(1, abs=1e-5)


def test_md_min():
    tol = 1e-8
    step = 0.1
    func = Paraboloid(pos=np.array([10, 10, 10], dtype=complex), shift=1.)
    md_min(func=func, step=step, tolerance=tol, verbose=False)
    assert func.get_functional_value() == pytest.approx(1, abs=1e-5)


def test_rotation_from_projection(rng):
    proj_nw = rng.rand(6, 4)
    assert orthonormality_error(proj_nw[:int(min(proj_nw.shape))]) > 1
    U_ww, C_ul = rotation_from_projection(proj_nw, fixed=2, ortho=True)
    assert orthonormality_error(U_ww) < 1e-10, 'U_ww not unitary'
    assert orthogonality_error(C_ul.T) < 1e-10, 'C_ul columns not orthogonal'
    assert normalization_error(C_ul) < 1e-10, 'C_ul not normalized'
    U_ww, C_ul = rotation_from_projection(proj_nw, fixed=2, ortho=False)
    assert normalization_error(U_ww) < 1e-10, 'U_ww not normalized'


def test_save(tmpdir, wan):
    wan = wan(nwannier=4, fixedstates=2, initialwannier='bloch')
    picklefile = tmpdir.join('wan.pickle')
    f1 = wan.get_functional_value()
    wan.save(picklefile)
    wan.initialize(file=picklefile, initialwannier='bloch')
    assert pytest.approx(f1) == wan.get_functional_value()


# The following test always fails because get_radii() is broken.
@pytest.mark.parametrize('lat', [CUB(1), FCC(1), BCC(1), TET(1, 2), BCT(1, 2),
                                 ORC(1, 2, 3), ORCF(1, 2, 3), ORCI(1, 2, 3),
                                 ORCC(1, 2, 3), HEX(1, 2), RHL(1, 110),
                                 MCL(1, 2, 3, 70), MCLC(1, 2, 3, 70),
                                 TRI(1, 2, 3, 60, 70, 80), OBL(1, 2, 110),
                                 HEX2D(1), RECT(1, 2), CRECT(1, 70), SQR(1),
                                 LINE(1)])
def test_get_radii(lat, wan):
    if ((lat.tocell() == FCC(a=1).tocell()).all() or
            (lat.tocell() == ORCF(a=1, b=2, c=3).tocell()).all()):
        pytest.skip("lattices not supported, yet")
    atoms = molecule('H2', pbc=True)
    atoms.cell = lat.tocell()
    atoms.center(vacuum=3.)
    wan = wan(nwannier=4, fixedstates=2, atoms=atoms, initialwannier='bloch')
    assert not (wan.get_radii() == 0).all()


def test_get_functional_value(wan):
    # Only testing if the functional scales with the number of functions
    wan1 = wan(nwannier=3)
    f1 = wan1.get_functional_value()
    wan2 = wan(nwannier=4)
    f2 = wan2.get_functional_value()
    assert f1 < f2


def test_get_centers():
    # Rough test on the position of the Wannier functions' centers
    gpaw = pytest.importorskip('gpaw')
    calc = gpaw.GPAW(gpts=(32, 32, 32), nbands=4, txt=None)
    atoms = molecule('H2', calculator=calc)
    atoms.center(vacuum=3.)
    atoms.get_potential_energy()
    wan = Wannier(nwannier=2, calc=calc, initialwannier='bloch')
    centers = wan.get_centers()
    com = atoms.get_center_of_mass()
    assert np.abs(centers - [com, com]).max() < 1e-4


def test_write_cube(wan):
    atoms = molecule('H2')
    atoms.center(vacuum=3.)
    wan = wan(atoms=atoms)
    index = 0
    # It returns some errors when using file objects, so we use simple filename
    cubefilename = 'wan.cube'
    wan.write_cube(index, cubefilename)
    with open(cubefilename, mode='r') as inputfile:
        content = read_cube(inputfile)
    assert pytest.approx(content['atoms'].cell.array) == atoms.cell.array
    assert pytest.approx(content['data']) == wan.get_function(index)


def test_localize(wan):
    wan = wan(initialwannier='random')
    fvalue = wan.get_functional_value()
    wan.localize()
    assert wan.get_functional_value() > fvalue


def test_get_spectral_weight_bloch(wan):
    nwannier = 4
    wan = wan(initialwannier='bloch', nwannier=nwannier)
    for i in range(nwannier):
        assert wan.get_spectral_weight(i)[:, i].sum() == pytest.approx(1)


def test_get_spectral_weight_random(wan, rng):
    nwannier = 4
    wan = wan(initialwannier='random', nwannier=nwannier, rng=rng)
    for i in range(nwannier):
        assert wan.get_spectral_weight(i).sum() == pytest.approx(1)


def test_get_pdos(wan):
    nwannier = 4
    gpaw = pytest.importorskip('gpaw')
    calc = gpaw.GPAW(gpts=(16, 16, 16), nbands=nwannier, txt=None)
    atoms = molecule('H2')
    atoms.center(vacuum=3.)
    atoms.calc = calc
    atoms.get_potential_energy()
    wan = wan(atoms=atoms, calc=calc, nwannier=nwannier, initialwannier='bloch')
    eig_n = calc.get_eigenvalues()
    for i in range(nwannier):
        pdos_n = wan.get_pdos(w=i, energies=eig_n, width=0.001)
        assert pdos_n[i] != pytest.approx(0)


def test_translate(wan):
    nwannier = 2
    atoms = molecule('H2', pbc=True)
    atoms.center(vacuum=3.)
    wan = wan(atoms=atoms, kpts=(2, 2, 2),
              nwannier=nwannier, initialwannier='bloch')
    wan.translate_all_to_cell(cell=[0, 0, 0])
    c0_n = wan.get_centers()
    for i in range(nwannier):
        c2_n = np.delete(wan.get_centers(), i, 0)
        wan.translate(w=i, R=[1, 1, 1])
        c1_n = wan.get_centers()
        assert np.linalg.norm(c1_n[i] - c0_n[i]) == \
            pytest.approx(np.linalg.norm(atoms.cell.array.diagonal()))
        c1_n = np.delete(c1_n, i, 0)
        assert c1_n == pytest.approx(c2_n)


def test_translate_to_cell(wan):
    nwannier = 2
    atoms = molecule('H2', pbc=True)
    atoms.center(vacuum=3.)
    wan = wan(atoms=atoms, kpts=(2, 2, 2),
              nwannier=nwannier, initialwannier='bloch')
    for i in range(nwannier):
        wan.translate_to_cell(w=i, cell=[0, 0, 0])
        c0_n = wan.get_centers()
        assert (c0_n[i] < atoms.cell.array.diagonal()).all()
        wan.translate_to_cell(w=i, cell=[1, 1, 1])
        c1_n = wan.get_centers()
        assert (c1_n[i] > atoms.cell.array.diagonal()).all()
        assert np.linalg.norm(c1_n[i] - c0_n[i]) == \
            pytest.approx(np.linalg.norm(atoms.cell.array.diagonal()))
        c0_n = np.delete(c0_n, i, 0)
        c1_n = np.delete(c1_n, i, 0)
        assert c0_n == pytest.approx(c1_n)


def test_translate_all_to_cell(wan):
    nwannier = 2
    atoms = molecule('H2', pbc=True)
    atoms.center(vacuum=3.)
    wan = wan(atoms=atoms, kpts=(2, 2, 2),
              nwannier=nwannier, initialwannier='bloch')
    wan.translate_all_to_cell(cell=[0, 0, 0])
    c0_n = wan.get_centers()
    assert (c0_n < atoms.cell.array.diagonal()).all()
    wan.translate_all_to_cell(cell=[1, 1, 1])
    c1_n = wan.get_centers()
    assert (c1_n > atoms.cell.array.diagonal()).all()
    for i in range(nwannier):
        assert np.linalg.norm(c1_n[i] - c0_n[i]) == \
            pytest.approx(np.linalg.norm(atoms.cell.array.diagonal()))
