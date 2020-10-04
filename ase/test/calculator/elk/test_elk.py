import pytest

from ase.build import bulk, molecule
from ase.units import Hartree


@pytest.mark.calculator('elk', tasks=0, ngridk=(5, 5, 5))
def test_elk_bulk(factory):
    atoms = bulk('Si')
    calc = factory.calc()
    atoms.calc = calc
    energy = atoms.get_potential_energy()
    assert energy == pytest.approx(-578.077659570 * Hartree, abs=0.1)
