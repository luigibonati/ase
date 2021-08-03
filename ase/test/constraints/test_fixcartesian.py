import pytest
import numpy as np
from ase.constraints import FixCartesian
from ase.build import molecule

@pytest.fixture
def atoms():
    return molecule('CH3CH2OH')


def test_fixcartesian_misc():
    mask = np.array([0, 1, 0], bool)
    constraint = FixCartesian(3, mask=mask)
    assert '3' in str(constraint)
    dct = constraint.todict()['kwargs']

    assert dct['a'] == 3
    assert all(dct['mask'] == mask)


def test_fixcartesian_adjust(atoms):
    cart_mask = np.array([False, True, True])
    atom_index = 3

    fixmask = np.zeros((len(atoms), 3), bool)
    fixmask[atom_index, cart_mask] = True

    oldpos = atoms.get_positions()
    constraint = FixCartesian(3, mask=cart_mask)

    rng = np.random.RandomState(42)
    deviation = 1.0 + rng.rand(len(atoms), 3)

    newpos = oldpos + deviation
    constraint.adjust_positions(atoms, newpos)

    newpos_expected = oldpos + deviation
    newpos_expected[fixmask] = oldpos[fixmask]

    assert newpos == pytest.approx(newpos_expected, abs=1e-14)

    oldforces = 1.0 + np.random.rand(len(atoms), 3)
    newforces = oldforces.copy()
    constraint.adjust_forces(atoms, newforces)

    newforces_expected = oldforces.copy()
    newforces_expected[fixmask] = 0.0
    assert newforces == pytest.approx(newforces_expected, abs=1e-14)
