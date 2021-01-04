import pytest
from ase import Atoms


@pytest.fixture
def atoms_co():
    """Simple atoms object for testing with a single CO molecule"""
    d = 1.14
    atoms = Atoms('CO', positions=[(0, 0, 0), (0, 0, d)], pbc=True)
    atoms.center(vacuum=5)
    return atoms


@pytest.fixture
def atoms_2co():
    """Simple atoms object for testing with 2x CO molecules"""
    d = 1.14
    atoms = Atoms('CO', positions=[(0, 0, 0), (0, 0, d)], pbc=True)
    atoms.extend(Atoms('CO', positions=[(0, 2, 0), (0, 2, d)]))

    atoms.center(vacuum=5.)
    return atoms
