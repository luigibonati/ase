import pytest
from ase.build import bulk
from ase.io.pickletrajectory import PickleTrajectory
from ase.calculators.calculator import compare_atoms


trajname = 'pickletraj.traj'


def test_pickletrajectory():
    atoms = bulk('Ti')
    atoms1 = atoms.copy()
    atoms1.rattle()
    images = [atoms, atoms1]

    ptraj = PickleTrajectory(trajname, 'w', _warn=False)
    for image in images:
        ptraj.write(image)
    ptraj.close()

    ptraj1 = PickleTrajectory(trajname, _warn=False)
    images1 = list(ptraj1)
    print(images1)
    assert len(images) == len(images1)
    for atoms, atoms1 in zip(images, images1):
        assert not compare_atoms(atoms, atoms1)
