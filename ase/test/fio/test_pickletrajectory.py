import pytest
import numpy as np
from ase.build import bulk
from ase.io.pickletrajectory import PickleTrajectory
from ase.calculators.calculator import compare_atoms
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms

trajname = 'pickletraj.traj'

def test_raises():
    with pytest.raises(DeprecationWarning):
        PickleTrajectory(trajname, 'w')


@pytest.fixture
def images():
    atoms = bulk('Ti')
    atoms.symbols = 'Au'
    atoms.calc = EMT()
    atoms1 = atoms.copy()
    atoms1.rattle()
    images = [atoms, atoms1]

    # Set all sorts of weird data:
    for i, atoms in enumerate(images):
        ints = np.arange(len(atoms)) + i
        floats = 1.0 + np.arange(len(atoms))
        atoms.set_tags(ints)
        atoms.set_initial_magnetic_moments(floats)
        atoms.set_initial_charges(floats)
        atoms.set_masses(floats)
        floats3d = 1.2 * np.arange(3 * len(atoms)).reshape(-1, 3)
        atoms.set_momenta(floats3d)
        atoms.info = {'hello': 'world'}
        atoms.calc = EMT()
        atoms.get_potential_energy()

    atoms.set_constraint(FixAtoms(indices=[0]))
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
