import pytest
from ase.io import read, write
from ase.build import bulk
from ase.calculators.calculator import compare_atoms
from ase.io.vasp_parsers.vasp_structure_io import (read_vasp_structure,
                                                   write_vasp_structure)

@pytest.fixture
def atoms():
    _atoms = bulk('NaCl', crystalstructure='rocksalt', a=4.1, cubic=True)
    _atoms.wrap()
    return _atoms


@pytest.mark.parametrize('filename', ['POSCAR', 'CONTCAR'])
@pytest.mark.parametrize('vasp5', [True, False])
def test_read_write_roundtrip(atoms, vasp5, filename):
    write(filename, atoms, vasp5=vasp5)
    atoms_loaded = read(filename)

    assert len(compare_atoms(atoms, atoms_loaded)) == 0


@pytest.mark.parametrize('filename', ['POSCAR', 'CONTCAR'])
@pytest.mark.parametrize('kwargs', [{}, {'vasp5': True}])
def test_write_vasp5(atoms, filename, kwargs):
    """Test that we write the symbols to the POSCAR/CONTCAR
    with vasp5=True (which should also be the default)"""
    write(filename, atoms, format='vasp', **kwargs)
    with open(filename) as file:
        lines = file.readlines()
    # Test the 5th line, which should be the symbols
    assert lines[5].strip().split() == list(atoms.symbols)


@pytest.mark.parametrize('filename', ['POSCAR', 'CONTCAR'])
@pytest.mark.parametrize('kwargs', [{'direct':[True, False]}, {'sort':[True,False]}, {'long_format':[True,False]}, {'vasp5': True}, {'ignore_constraints':[True, False]},{'wrap': [True, False]}])
def test_write_poscar(atoms, filename, kwargs):
    write_vasp_structure(filename,
                         atoms=atoms,
                         label=None,
                         direct=False,
                         sort=True,
                         symbol_count=None,
                         long_format=True,
                         vasp5=True,
                         ignore_constraints=False,
                         wrap=False)
    res = ['Cl Na\n',
           '  1.0000000000000000\n',
           '    4.0999999999999996    0.0000000000000000    0.0000000000000000\n',
           '    0.0000000000000000    4.0999999999999996    0.0000000000000000\n',
           '    0.0000000000000000    0.0000000000000000    4.0999999999999996\n',
           '  Cl Na\n',
           '  4  4\n',
           'Cartesian\n',
           '  2.0499999999999994  0.0000000000000000  0.0000000000000000\n',
           '  2.0499999999999994  2.0499999999999994  2.0499999999999994\n',
           '  0.0000000000000000  0.0000000000000000  2.0499999999999994\n',
           '  0.0000000000000000  2.0499999999999994  0.0000000000000000\n',
           '  0.0000000000000000  0.0000000000000000  0.0000000000000000\n',
           '  0.0000000000000000  2.0499999999999994  2.0499999999999994\n',
           '  2.0499999999999994  0.0000000000000000  2.0499999999999994\n',
           '  2.0499999999999994  2.0499999999999994  0.0000000000000000\n']
    with open(filename) as fil:
        for i, line in enumerate(fil.readlines()):
            for j, elem in enumerate(line.split()):
                assert elem == res[i].split()[j]
    with open(filename) as fil:
        pass
        #assert fil.read() == res
    read_vasp_structure(filename)
