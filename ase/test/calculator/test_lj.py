import numpy as np
import pytest

from ase import Atoms
from ase.build import bulk
from ase.calculators.lj import LennardJones


# test molecular properties
reference_potential_energy = pytest.approx(-1.0)


@pytest.fixture
def atoms():
    """two atoms at potential minimum"""

    atoms = Atoms('H2', positions=[[0, 0, 0], [0, 0, 2 ** (1.0 / 6.0)]])
    calc = LennardJones(rc=1.0e5)
    atoms.calc = calc

    return atoms


def test_minimum_energy(atoms):
    # testing at the minimum to see if anything is on fire
    # See https://en.wikipedia.org/wiki/Lennard-Jones_potential
    # Minimum is at r=2^(1/6)*sigma, and it's -1.

    energy = atoms.get_potential_energy()
    energies = atoms.get_potential_energies()

    assert energy == reference_potential_energy
    assert energies.sum() == reference_potential_energy


def test_minimum_forces(atoms):
    # forces should be zero
    forces = atoms.get_forces()
    np.testing.assert_allclose(forces, 0, atol=1e-14)


def test_system_changes(atoms):
    # https://gitlab.com/ase/ase/-/merge_requests/1817
    atoms.calc.calculate(atoms, system_changes=['positions'])

    assert atoms.get_potential_energy() == reference_potential_energy


# test bulk properties
reference_force = pytest.approx(1.57190846e-05)
reference_pressure = pytest.approx(1.473229212e-05)


@pytest.fixture
def atoms_bulk():
    """return a stretched Ar fcc cell"""

    stretch = 1.5

    atoms = bulk("Ar", cubic=True)
    atoms.set_cell(atoms.cell * stretch, scale_atoms=True)

    calc = LennardJones(rc=10)
    atoms.calc = calc

    return atoms


def test_bulk_energies(atoms_bulk):
    # check energies
    energy = atoms_bulk.get_potential_energy()
    energies = atoms_bulk.get_potential_energies()

    assert np.allclose(energy, energies.sum())

    # energies should be equal in this high-symmetry structure
    assert energies.std() == pytest.approx(0.0)


def test_bulk_forces(atoms_bulk):
    # displace atom for 0.03 \AA
    atoms_bulk.positions[0, 0] += 0.03

    # check forces sum to zero
    forces = atoms_bulk.get_forces()

    assert np.allclose(forces.sum(axis=0), 0)

    # check reference force
    assert forces[0, 0] == reference_force


def test_bulk_stress(atoms_bulk):
    # check stress computation for sanity and reference
    stress = atoms_bulk.get_stress()
    stresses = atoms_bulk.get_stresses()

    assert np.allclose(stress, stresses.sum(axis=0))

    # check reference pressure
    pressure = sum(stress[:3]) / 3

    assert pressure == reference_pressure
