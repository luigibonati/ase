"""
Use lammpsdata module to create an Atoms object from a lammps data file
and checks that the cell, mass, positions, and velocities match the
values that are parsed directly from the data file.

NOTE: This test currently only works when using a lammps data file
containing a single atomic species
"""
import pytest
import ase.io

from .parse_lammps_data_file import lammpsdata_file_extracted_sections

# Relative tolerance for comparing floats with pytest.approx
REL_TOL = 1e-2


def test_lammpsdata_read(lammpsdata_file_path):
    atoms = ase.io.read(lammpsdata_file_path, format="lammps-data", units="metal")

    expected_values = lammpsdata_file_extracted_sections(lammpsdata_file_path)

    # Check cell was read in correctly
    cell_read_in = atoms.get_cell()
    for cell_vec_read_in, cell_vec_expected in zip(
        cell_read_in, expected_values["cell"]
    ):
        assert cell_vec_read_in == pytest.approx(cell_vec_expected, rel=REL_TOL)

    # Check masses were read in correctly
    masses_read_in = atoms.get_masses()
    masses_expected = [expected_values["mass"]] * len(expected_values["positions"])
    assert masses_read_in == pytest.approx(masses_expected, rel=REL_TOL)

    # Check positions were read in correctly
    positions_read_in = atoms.get_positions()
    positions_expected = expected_values["positions"]
    for pos_read_in, pos_expected in zip(positions_read_in, positions_expected):
        assert pos_read_in == pytest.approx(pos_expected, rel=REL_TOL)

    # Check velocities were read in correctly
    velocities_read_in = atoms.get_velocities()
    velocities_expected = expected_values["velocities"]
    for vel_read_in, vel_expected in zip(velocities_read_in, velocities_expected):
        assert vel_read_in == pytest.approx(vel_expected, rel=REL_TOL)

    # TODO: Also check charges, travels, molecule id, bonds, and angles
