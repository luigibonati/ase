# import inspect
import pytest
import numpy as np
from shutil import copyfile
from ase.io import read


@pytest.fixture
def vasprun(datadir):
    return datadir / 'vasp' / 'vasprun.xml'


def test_atoms(vasprun, tmp_path):

    copyfile(vasprun, tmp_path / 'vasprun.xml')
    atoms = read(tmp_path / 'vasprun.xml', index=-1)

    # check number of atoms
    assert len(atoms) == 2

    # make sure it is still tungsten
    assert all(np.array(atoms.get_chemical_symbols()) == "W")

    # check scaled_positions
    expected_scaled_positions = np.array([[0.0, 0.0, 0.0],
                                         [0.5,  0.5, 0.5]])

    np.testing.assert_array_equal(atoms.get_scaled_positions(),
                                  expected_scaled_positions)

    expected_cell = np.array([[3.16, 0.0, 0.0],
                              [0.0, 3.16, 0.0],
                              [0.0, 0.0, 3.16]])

    # check cell
    np.testing.assert_array_equal(atoms.cell, expected_cell)

    # check real positions
    np.testing.assert_array_equal(atoms.positions,
                                  expected_scaled_positions @
                                  atoms.cell.complete())


def test_calculation(vasprun, tmp_path):

    from ase.units import GPa

    copyfile(vasprun, tmp_path / 'vasprun.xml')
    atoms = read(tmp_path / 'vasprun.xml', index=-1)

    expected_e_0_energy = -29.67691672
    assert atoms.get_potential_energy() == expected_e_0_energy

    expected_e_fr_energy = -29.67243317
    assert (atoms.get_potential_energy(force_consistent=True) ==
            expected_e_fr_energy)

    expected_forces = np.array([[7.58587457, -5.22590317, 6.88227285],
                                [-7.58587457, 5.22590317, -6.88227285]])
    np.testing.assert_array_equal(atoms.get_forces(),
                                  expected_forces)

    expected_stress = np.array([[4300.36902090, -284.50040544, -1468.20603140],
                               [-284.50040595, 4824.17435683, -1625.37541639],
                               [-1468.20603158, -1625.37541697, 5726.84189498]])

    expected_stress *= -0.1 * GPa
    expected_stress = expected_stress.reshape(9)[[0, 4, 8, 5, 2, 1]]

    np.testing.assert_array_equal(atoms.get_stress(), expected_stress)

    expected_kpoints = np.array([[0.0, 0.0, 0.0]])
    np.testing.assert_array_equal(atoms.calc.ibz_kpts, expected_kpoints)