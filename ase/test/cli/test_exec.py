import pytest
from ase.build import bulk
from ase.io import write


@pytest.fixture
def atoms():
    return bulk('Au')


@pytest.fixture
def fname(atoms):
    filename = 'file.traj'
    write(filename, atoms)
    return filename


def test_exec_fail_withoutcode(cli, fname):
    cli.ase('exec', fname, expect_fail=True)


def test_exec_atoms(cli, fname, atoms):
    out = cli.ase('exec', fname, '-e', 'print(atoms.symbols)')
    assert out.strip() == str(atoms.symbols)


def test_exec_index(cli, fname):
    out = cli.ase('exec', fname, '-e', 'print(index)')
    assert out.strip() == str(0)


def test_exec_images(cli, fname, atoms):
    out = cli.ase('exec', fname, '-e', 'print(len(images[0]))')
    assert out.strip() == str(len(atoms))
