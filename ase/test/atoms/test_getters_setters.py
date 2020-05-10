import pytest
import numpy as np
from ase.build import molecule


@pytest.fixture
def atoms():
    return molecule('CH3CH2OH')


def test_numbers(atoms):
    numbers = range(len(atoms))
    atoms.numbers = numbers
    assert all(atoms.numbers == numbers)

    new_numbers = atoms.numbers + 1
    atoms.set_atomic_numbers(new_numbers)
    assert all(atoms.get_atomic_numbers() == new_numbers)


def test_positions(atoms):
    positions = [range(3)] * len(atoms)
    atoms.positions = positions
    assert (atoms.positions == positions).all()

    new_positions = atoms.positions + 1
    atoms.set_positions(new_positions)
    assert atoms.get_positions() == pytest.approx(new_positions)
