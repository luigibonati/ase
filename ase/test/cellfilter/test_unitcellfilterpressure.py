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
