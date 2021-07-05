import numpy as np
import pytest

from ase.build.bulk import bulk
from ase.cluster import Icosahedron
from ase.calculators.emt import EMT
from ase.optimize.fire import FIRE
from ase.lattice.compounds import L1_2
import timeit

from ase.ga.utilities import get_rdf, CellIsNotLargeEnoughError, get_recommended_r_max


def test_rdf_performance():
    atoms = Icosahedron('Cu', 4)
    atoms.center(vacuum=0.0)
    rdf, dists = get_rdf(atoms, 8.0, 6)
    rdf_ref = np.array([0., 0.81280406995,  0.716085965517, 0.835726863932,
                        0.40215359365, 0.42679009693 ])
    np.set_printoptions(precision=12)
    dists_ref = np.array([0.66666667,  2., 3.33333333, 4.66666667, 6.,7.33333333])
    assert dists == pytest.approx(dists_ref)
    assert rdf == pytest.approx(rdf_ref)
    t100 = timeit.timeit(lambda: get_rdf(atoms, 8.0, 60, no_dists=True), number=100)
    print(t100)

def test_rdf_exceptions():
    with pytest.raises(CellIsNotLargeEnoughError):
        get_rdf(bulk('Ag'), 2.0, 5)


def test_rdf_compute():
    eps = 1e-5

    atoms = Icosahedron('Cu', 3)
    atoms.center(vacuum=0.0)
    atoms.numbers[[0, 13, 15, 16, 18, 19, 21, 22, 24, 25, 27, 28, 30]] = 79
    atoms.calc = EMT()
    with FIRE(atoms, logfile=None) as opt:
        opt.run(fmax=0.05)

    rmax = 8.
    nbins = 5
    rdf, dists = get_rdf(atoms, rmax, nbins)
    calc_dists = np.arange(rmax / (2 * nbins), rmax, rmax / nbins)
    assert all(abs(dists - calc_dists) < eps)
    reference_rdf1 = [0., 0.84408157, 0.398689, 0.23748934, 0.15398546]
    assert all(abs(rdf - reference_rdf1) < eps)

    dm = atoms.get_all_distances()
    s = np.zeros(5)
    for c in [(29, 29), (29, 79), (79, 29), (79, 79)]:
        inv_norm = len(np.where(atoms.numbers == c[0])[0]) / len(atoms)
        s += get_rdf(atoms, rmax, nbins, elements=c,
                     distance_matrix=dm, no_dists=True) * inv_norm
    assert all(abs(s - reference_rdf1) < eps)

    AuAu = get_rdf(atoms, rmax, nbins, elements=(79, 79),
                   distance_matrix=dm, no_dists=True)
    assert all(abs(AuAu[-2:] - [0.12126445, 0.]) < eps)

    bulk = L1_2(['Au', 'Cu'], size=(3, 3, 3), latticeconstant=2 * np.sqrt(2))
    rdf = get_rdf(bulk, 4.2, 5)[0]
    reference_rdf2 = [0., 0., 1.43905094, 0.36948605, 1.34468694]
    assert all(abs(rdf - reference_rdf2) < eps)
