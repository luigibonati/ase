import pytest
from ase.build import molecule


@pytest.fixture
def atoms():
    return molecule('NH3')


@pytest.fixture
def magmoms(atoms):
    return atoms.magmoms


def test_unpol(magmoms):
    assert not magmoms
    assert magmoms.spincomponents == 1
    assert magmoms.spin_type == 'paired'
    assert not magmoms.polarized
    assert magmoms.collinear


def test_pol(atoms, magmoms):
    atoms.set_initial_magnetic_moments(range(len(atoms)))
    assert magmoms
    assert magmoms.spincomponents == 2
    assert magmoms.spin_type == 'collinear'
    assert magmoms.polarized
    assert magmoms.collinear


def test_noncollinear(atoms, magmoms):
    atoms.set_initial_magnetic_moments([[1, 2, 3]] * len(atoms))
    assert magmoms
    assert magmoms.spincomponents == 3
    assert magmoms.spin_type == 'noncollinear'
    assert magmoms.polarized
    assert not magmoms.collinear


def test_atoms_magmoms_property(atoms):
    magmoms = atoms.magmoms
    assert not magmoms.polarized
    atoms.set_initial_magnetic_moments(range(len(atoms)))
    assert magmoms.polarized
    assert magmoms.collinear
