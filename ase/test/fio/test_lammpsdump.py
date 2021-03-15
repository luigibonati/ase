import numpy as np
import pytest
from ase.io.formats import ioformats, match_magic


def lammpsdump_headers():
    actual_magic = 'ITEM: TIMESTEP'
    yield actual_magic
    yield f'anything\n{actual_magic}\nanything'


@pytest.mark.parametrize('header', lammpsdump_headers())
def test_recognize_lammpsdump(header):
    fmt_name = 'lammps-dump-text'
    fmt = match_magic(header.encode('ascii'))
    assert fmt.name == fmt_name


def test_lammpsdump_xy_yu_zu():
    # Test lammpsdump with positions given as xu, yu, zu
    buf = """\
ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
3
ITEM: BOX BOUNDS pp pp pp
0.0e+00 4e+00
0.0e+00 5.0e+00
0.0e+00 2.0e+01
ITEM: ATOMS id type xu yu zu
1 1 1.2 1.3 1.4
3 1 2.3 2.4 2.5
2 1 3.4 3.5 3.6
"""

    ref_positions = np.array([[1.2, 1.3, 1.4],
                              [2.3, 2.4, 2.5],
                              [3.4, 3.5, 3.6]])
    order = np.array([1, 3, 2])

    fmt = ioformats['lammps-dump-text']
    atoms = fmt.parse_atoms(buf)

    assert atoms.cell.orthorhombic
    assert pytest.approx(atoms.cell.lengths()) == [4., 5., 20.]
    assert pytest.approx(atoms.positions) == ref_positions[order - 1]
