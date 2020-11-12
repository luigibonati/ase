import pytest
import numpy as np
from ase import Atoms, Atom
from ase.constraints import Hookean, FixAtoms


@pytest.fixture
def atoms():
    atoms = Atoms('H2')
    atoms.positions[0, 2] = 2.0
    return atoms


def test_momenta_fixatoms(atoms):
    atoms.set_constraint(FixAtoms(indices=[0]))
    atoms.set_momenta(np.ones(atoms.get_momenta().shape))
    desired = np.ones(atoms.get_momenta().shape)
    desired[0] = 0.
    actual = atoms.get_momenta()
    assert (actual == desired).all()


def test_momenta_hookean(atoms):
    atoms.set_constraint(Hookean(0, 1, rt=1., k=10.))
    atoms.set_momenta(np.zeros(atoms.get_momenta().shape))
    actual = atoms.get_momenta()
    desired = np.zeros(atoms.get_momenta().shape)
    assert (actual == desired).all()
