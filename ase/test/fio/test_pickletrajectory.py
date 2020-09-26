import pytest
from ase.build import bulk
from ase.io.pickletrajectory import PickleTrajectory
from ase.calculators.calculator import compare_atoms


trajname = 'pickletraj.traj'

def test_raises():
    with pytest.raises(DeprecationWarning):
        PickleTrajectory(trajname, 'w')


@pytest.fixture
def images():
    atoms = bulk('Ti')
    atoms1 = atoms.copy()
    atoms1.rattle()
    return [atoms, atoms1]


def read_images(filename):
    with PickleTrajectory(filename, _warn=False) as traj:
        return list(traj)


@pytest.fixture
def trajfile(images):
    ptraj = PickleTrajectory(trajname, 'w', _warn=False)
    for image in images:
        ptraj.write(image)
    ptraj.close()
    return trajname


def test_write_read(images, trajfile):
    images1 = read_images(trajfile)
    assert len(images) == len(images1)
    for atoms, atoms1 in zip(images, images1):
        assert not compare_atoms(atoms, atoms1)


def test_append(images, trajfile):
    with PickleTrajectory(trajfile, 'a', _warn=False) as traj:
        for image in images:
            traj.write(image)

    images1 = read_images(trajfile)
    assert len(images1) == 4
    for atoms, atoms1 in zip(images * 2, images1):
        assert not compare_atoms(atoms, atoms1)
