import numpy as np
import pytest

from ase.units import GPa
from ase.build import bulk
from ase.calculators.test import gradient_test
from ase.constraints import UnitCellFilter, ExpCellFilter
from ase.optimize import FIRE, LBFGSLineSearch


@pytest.fixture
def atoms(asap3):
    rng = np.random.RandomState(0)
    atoms = bulk('Cu', cubic=True)

    # Perturb:
    atoms.positions[:, 0] *= 0.995
    atoms.cell += rng.uniform(-1e-2, 1e-2, size=9).reshape((3,3))

    atoms.calc = asap3.EMT()
    return atoms


@pytest.mark.parametrize('cellfilter', [UnitCellFilter, ExpCellFilter])
def test_pressure(atoms, cellfilter):
    xcellfilter = cellfilter(atoms, scalar_pressure=10.0 * GPa)

    # test all derivatives
    f, fn = gradient_test(xcellfilter)
    assert abs(f - fn).max() < 5e-6

    opt = LBFGSLineSearch(xcellfilter)
    opt.run(1e-3)

    # check pressure is within 0.1 GPa of target
    sigma = atoms.get_stress() / GPa
    pressure = -(sigma[0] + sigma[1] + sigma[2]) / 3.0
    assert abs(pressure - 10.0) < 0.1


def test_cellfilter(atoms):
    ucf = UnitCellFilter(atoms)
    f, fn = gradient_test(ucf)
    assert abs(f - fn).max() < 3e-6


def test_expcellfilter(atoms):
    ecf = ExpCellFilter(atoms)
    # test all derivatives
    f, fn = gradient_test(ecf)
    assert abs(f - fn).max() < 3e-6


from math import sqrt
from ase import Atoms
from ase.optimize import LBFGS
from ase.constraints import UnitCellFilter
from ase.io import Trajectory
from ase.optimize.mdmin import MDMin


# XXX This test should have some assertions!  --askhl
def test_unitcellfilter(asap3):
    a = 3.6
    b = a / 2
    cu = Atoms('Cu',
               cell=[(0, b, b), (b, 0, b), (b, b, 0)],
               pbc=1) * (6, 6, 6)
    cu.calc = asap3.EMT()
    f = UnitCellFilter(cu, [1, 1, 1, 0, 0, 0])
    opt = LBFGS(f)
    t = Trajectory('Cu-fcc.traj', 'w', cu)
    opt.attach(t)
    opt.run(5.0)

    # HCP:
    from ase.build import bulk
    cu = bulk('Cu', 'hcp', a=a / sqrt(2))
    cu.cell[1,0] -= 0.05
    cu *= (6, 6, 3)
    cu.calc = asap3.EMT()
    print(cu.get_forces())
    print(cu.get_stress())
    f = UnitCellFilter(cu)
    opt = MDMin(f, dt=0.01)
    t = Trajectory('Cu-hcp.traj', 'w', cu)
    opt.attach(t)
    opt.run(0.2)
