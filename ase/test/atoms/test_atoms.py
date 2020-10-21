import pytest
import numpy as np
from ase import Atoms


def test_atoms():
    from ase import Atoms
    print(Atoms())
    print(Atoms('H2O'))
    #...


def test_numbers_input():
    numbers= np.array([[0, 1], [2, 3]])
    with pytest.raises(Exception, match='"numbers" must be 1-dimensional.'):
        Atoms(positions=np.zeros((2, 3)), numbers=numbers, cell=np.eye(3))

    Atoms(positions=np.zeros((2, 3)), numbers=[0, 1], cell=np.eye(3))


def test_bad_array_shape():
    with pytest.raises(ValueError, match='wrong length'):
        Atoms().set_masses([1, 2])

    with pytest.raises(ValueError, match='wrong length'):
        Atoms('H').set_masses([])

    with pytest.raises(ValueError, match='wrong shape'):
        Atoms('H').set_masses(np.ones((1, 3)))


def test_set_masses():
    atoms = Atoms('AgAu')
    m0 = atoms.get_masses()
    atoms.set_masses([1, None])
    assert atoms.get_masses() == pytest.approx([1, m0[1]])
