import pytest

from ase import Atoms
from ase.data.refstate import BulkReferenceState, define_reference_state
from ase.calculators.calculator import compare_atoms


Z = 42
symbol = 'Mo'


def test_basics():
    dct = dict(oranges=5, potatoes=7, symmetry='atom')
    state = define_reference_state(Z, **dct)
    assert state.symbol == symbol
    assert state.Z == Z
    assert 'oranges=5' in str(state)

    # Check the Mapping methods:
    assert list(state) == list(dct)
    assert len(state) == len(dct)
    assert dict(state) == dct


def test_atom():
    state = define_reference_state(Z, symmetry='atom')
    print(state)

    atoms = state.toatoms()
    assert not compare_atoms(atoms, Atoms([Z]))


def test_dimer():
    d0 = 0.8
    state = define_reference_state(Z, symmetry='diatom', d=d0)
    assert state.bondlength == pytest.approx(d0)
    print(state)

    atoms = state.toatoms()

    assert len(atoms) == 2
    assert pytest.approx(atoms.positions[:, :2]) == 0
    assert pytest.approx(atoms.positions[:, 2]) == [-d0 / 2, d0 / 2]
    assert all(atoms.numbers == Z)


def test_bulk_fcc():
    state = define_reference_state(Z, symmetry='fcc', a=4.0, sgs=225)
    assert int(state.spacegroup) == 225
    assert state.lattice_centering == 'F'  # face-centered'
    assert state.crystal_family == 'c'  # cubic
