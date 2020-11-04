import pytest
from ase import Atoms
from ase.io import read, write
from ase.build import molecule


def test_exciting_io():
    atoms = Atoms('N3O',
                  cell=[3, 4, 5],
                  positions=[(0, 0, 0), (1, 0, 0),
                             (0, 0, 1), (0.5, 0.5, 0.5)],
                  pbc=True)

    write('input.xml', atoms)
    atoms2 = read('input.xml')

    assert all(atoms.symbols == atoms2.symbols)
    assert atoms.cell[:] == pytest.approx(atoms2.cell[:])
    assert atoms.positions == pytest.approx(atoms2.positions)

    #Exciting(dir='excitingtestfiles',
    #         kpts=(4, 4, 3),
             # bin='/fshome/chm/git/exciting/bin/excitingser',
    #         maxscl=3)
    # maybe do something???
