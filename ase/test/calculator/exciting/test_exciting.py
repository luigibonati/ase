import pytest
from ase import Atoms
from ase.build import bulk
from ase.io import read, write


def test_exciting():
    a = Atoms('N3O',
              [(0, 0, 0), (1, 0, 0), (0, 0, 1), (0.5, 0.5, 0.5)],
              pbc=True)

    write('input.xml', a)
    b = read('input.xml')

    print(a)
    print(a.get_positions())
    print(b)
    print(b.get_positions())


@pytest.mark.calculator_lite
@pytest.mark.calculator('exciting')
def test_exciting_bulk(factory):
    atoms = bulk('Si')
    atoms.calc = factory.calc()
    energy = atoms.get_potential_energy()
    print(energy)
