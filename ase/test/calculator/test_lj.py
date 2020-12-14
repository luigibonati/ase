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


@pytest.fixture
def atoms_smooth():
    """two atoms at potential minimum, with 'smooth' LJ"""

    atoms = Atoms('H2', positions=[[0, 0, 0], [0, 0, 2 ** (1.0 / 6.0)]])
    calc = LennardJones(rc=1.0e5, smooth=True)
    atoms.calc = calc

    return atoms


def test_minimum_energy(atoms, atoms_smooth):
    # testing at the minimum to see if anything is on fire
    # See https://en.wikipedia.org/wiki/Lennard-Jones_potential
    # Minimum is at r=2^(1/6)*sigma, and it's -1.

    assert atoms.get_potential_energy() == reference_potential_energy
    assert atoms_smooth.get_potential_energy() == reference_potential_energy
    assert atoms.get_potential_energies().sum() == reference_potential_energy
    assert atoms_smooth.get_potential_energies().sum() == reference_potential_energy


def test_minimum_forces(atoms, atoms_smooth):
    # forces should be zero
    np.testing.assert_allclose(atoms.get_forces(), 0, atol=1e-14)
    np.testing.assert_allclose(atoms_smooth.get_forces(), 0, atol=1e-14)


def test_system_changes(atoms):
    # https://gitlab.com/ase/ase/-/merge_requests/1817
    atoms.calc.calculate(atoms, system_changes=['positions'])

    assert atoms.get_potential_energy() == reference_potential_energy


def test_finite_difference():
    """Ensure that we got the modified forces right."""
    h = 1e-10
    r = 8.0
    calc = LennardJones(smooth=True, ro=6, rc=10, sigma=3)
    atoms = Atoms('H2', positions=[[0, 0, 0], [r, 0, 0]])
    atoms2 = Atoms('H2', positions=[[0, 0, 0], [r + h, 0, 0]])
    atoms.calc = calc
    atoms2.calc = calc

    fd_force = (atoms2.get_potential_energy() - atoms.get_potential_energy()) / h
    force = atoms.get_forces()[0, 0]

    np.testing.assert_allclose(fd_force, force)


# test bulk properties
stretch = 1.5
reference_force = pytest.approx(1.57190846e-05)
reference_pressure = pytest.approx(1.473229212e-05)


@pytest.fixture
def atoms_bulk():
    """return a stretched Ar fcc cell"""
    atoms = bulk("Ar", cubic=True)
    atoms.set_cell(atoms.cell * stretch, scale_atoms=True)

    calc = LennardJones(rc=10)
    atoms.calc = calc

    return atoms


@pytest.fixture
def atoms_smooth_bulk():
    """return a stretched Ar fcc cell, with a 'smooth' LJ"""
    atoms = bulk("Ar", cubic=True)
    atoms.set_cell(atoms.cell * stretch, scale_atoms=True)

    # somewhat hand-picked parameters, but ok for comparison
    calc = LennardJones(rc=12, ro=10, smooth=True)
    atoms.calc = calc

    return atoms


def test_bulk_energies(atoms_bulk, atoms_smooth_bulk):
    # check energies

    assert np.allclose(
        atoms_bulk.get_potential_energy(), atoms_bulk.get_potential_energies().sum()
    )
    assert np.allclose(
        atoms_smooth_bulk.get_potential_energy(),
        atoms_smooth_bulk.get_potential_energies().sum(),
    )

    # energies should be equal in this high-symmetry structure
    assert atoms_bulk.get_potential_energies().std() == pytest.approx(0.0)
    assert atoms_smooth_bulk.get_potential_energies().std() == pytest.approx(0.0)


def test_bulk_forces(atoms_bulk, atoms_smooth_bulk):
    # displace atom for 0.03 \AA
    atoms_bulk.positions[0, 0] += 0.03
    atoms_smooth_bulk.positions[0, 0] += 0.03

    # check forces sum to zero
    assert np.allclose(atoms_bulk.get_forces().sum(axis=0), 0)
    assert np.allclose(atoms_smooth_bulk.get_forces().sum(axis=0), 0)

    # check reference force
    assert atoms_bulk.get_forces()[0, 0] == reference_force
    assert atoms_smooth_bulk.get_forces()[0, 0] == reference_force


def test_bulk_stress(atoms_bulk):
    # check stress computation for sanity and reference
    # reference value computed for "non-smooth" LJ
    stress = atoms_bulk.get_stress()
    stresses = atoms_bulk.get_stresses()

    assert np.allclose(stress, stresses.sum(axis=0))

    # check reference pressure
    pressure = sum(stress[:3]) / 3

    assert pressure == reference_pressure
