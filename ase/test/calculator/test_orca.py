import pytest
import numpy as np

from ase.calculators.orca import ORCA
from ase.atoms import Atoms
from ase.units import Hartree
from ase.optimize import BFGS


@pytest.fixture
def water():
    return Atoms('OHH', positions=[(0, 0, 0), (1, 0, 0), (0, 1, 0)])


def test_orca(water):
    water.calc = ORCA(label='water', orcasimpleinput='BLYP def2-SVP Engrad')

    with BFGS(water) as opt:
        opt.run(fmax=0.05)

    final_energy = water.get_potential_energy()
    np.testing.assert_almost_equal(final_energy, -2077.24420, decimal=0)


def test_orca_sp(water):
    water.calc = ORCA(label='water', orcasimpleinput='BLYP def2-SVP', task="SP")

    final_energy = water.get_potential_energy()
    np.testing.assert_almost_equal(final_energy, -2077.24420, decimal=0)


def test_orca_use_last_energy(water):
    water.calc = ORCA(
        label='water',
        orcasimpleinput='PBE def2-SVP Opt TightOpt')
    energy = water.get_potential_energy() / Hartree

    np.testing.assert_almost_equal(energy, -76.272686944630, decimal=6)
