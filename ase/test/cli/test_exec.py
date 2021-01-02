import pytest
from ase.build import bulk
from ase.io import write

@pytest.fixture
def fname():
    atoms = bulk('Au')
    filename = 'file.traj'
    write(filename, atoms)
    return filename

def test_exec_fail_withoutcode(cli, fname):
    cli.ase('exec', fname, expect_fail=True)

def test_exec_atoms(cli, fname):
    atoms = bulk('Au')
    out = cli.ase('exec', fname, '-e', 'print(atoms.symbols)')
    assert str(atoms.symbols) in out

def test_exec_index(cli, fname):
    out = cli.ase('exec', fname, '-e', 'print(index)')
    assert str(0) in out

def test_exec_images(cli, fname):
    atoms = bulk('Au')
    out = cli.ase('exec', fname, '-e', 'print(len(images[0]))')
    assert str(len(atoms)) in out
