import pytest

from ase.build import bulk

@pytest.mark.calculator('elk')
def test_elk(factory):
    atoms = bulk('Si')
    calc = factory.calc(tasks=0)
    atoms.calc = calc
    atoms.get_potential_energy()
