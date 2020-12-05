"""
Use lammpsdata module to create an Atoms object from a lammps data file and
checks that the cell, mass, positions, and velocities match the values that are
parsed directly from the data file.

NOTE: This test currently only works when using  a lammps data file containing
a single atomic species
"""
import re

import pytest
import numpy as np
import ase.io
from ase.calculators.lammps import convert


@pytest.fixture
def lammpsdata_file_path(datadir):
    return datadir / "lammpsdata_input.data"


def extract_cell(raw_datafile_contents):
    RE_CELL = re.compile(
        r"""
            ([0-9e+.-]+)\ ([0-9e+.-]+)\ xlo\ xhi\n
            ([0-9e+.-]+)\ ([0-9e+.-]+)\ ylo\ yhi\n
            ([0-9e+.-]+)\ ([0-9e+.-]+)\ zlo\ zhi\n
        """,
        flags=re.VERBOSE,
    )
    xlo, xhi, ylo, yhi, zlo, zhi = map(
        float, RE_CELL.search(raw_datafile_contents).groups()
    )

    cell = [[xhi - xlo, 0, 0], [0, yhi - ylo, 0], [0, 0, zhi - zlo]]

    return cell


def extract_mass(raw_datafile_contents):
    """
    NOTE: Assumes that only a single atomic species is present
    """
    RE_MASS = re.compile(
        r"""
            Masses\n
            .*?\n
            [0-9]\ ([0-9e+.-]+)\n
            \nAtoms
        """,
        flags=re.VERBOSE | re.DOTALL,
    )
    mass = float(RE_MASS.search(raw_datafile_contents).group(1))

    return mass


def extract_atom_quantities(raw_datafile_contents):
    # Grab all atoms lines
    RE_POSITIONS_BLOCK = re.compile(
        r"""
            Atoms\n
            .*?\n
            (.*)
            .*?\n\n
            Velocities
        """,
        flags=re.VERBOSE | re.DOTALL,
    )
    positions_block = RE_POSITIONS_BLOCK.search(raw_datafile_contents).group(1)

    # Now parse each individual atoms line for quantities
    charges = []
    positions = []
    travels = []

    for atom_line in positions_block.splitlines():
        RE_ATOM_LINE = re.compile(
            "[0-9]+ [0-9]+ [0-9]+ ([0-9e+.-]+) "
            "([0-9e+.-]+) ([0-9e+.-]+) ([0-9e+.-]+) "
            "([0-9-]+) ([0-9-]+) ([0-9-]+)"
        )
        q, x, y, z, *travel = RE_ATOM_LINE.match(atom_line).groups()
        charges.append(float(q))
        positions.append(list(map(float, [x, y, z])))
        travels.append(list(map(int, travel)))

    return charges, positions, travels


def extract_velocities(raw_datafile_contents):
    """
    NOTE: Assumes metal units are used in data file
    """
    # Grab all velocities lines
    RE_VELOCITIES_BLOCK = re.compile(
        r"""
            Velocities\n
            .*?\n
            (.*)
            .*?\n\n
            Bonds
        """,
        flags=re.VERBOSE | re.DOTALL,
    )
    velocities_block = RE_VELOCITIES_BLOCK.search(raw_datafile_contents).group(1)

    # Now parse each individual line for velocity
    velocities = []
    for velocities_line in velocities_block.splitlines():
        RE_VELOCITY = re.compile("[0-9]+ ([0-9e+.-]+) ([0-9e+.-]+) " "([0-9e+.-]+)")
        v = RE_VELOCITY.match(velocities_line).groups()
        velocities.append(list(map(float, v)))

    # Convert to ASE's velocity units (uses unusual unit of time)
    velocities = convert(np.asarray(velocities), "velocity", "metal", "ASE")

    return velocities


def lammpsdata_file_extracted_groups(lammpsdata_file_path):
    """
    Manually read the lammpsdata input file and grep for the different
    quantities we want to check
    """
    with open(lammpsdata_file_path) as f:
        raw_datafile_contents = f.read()

    cell = extract_cell(raw_datafile_contents)
    mass = extract_mass(raw_datafile_contents)
    charges, positions, travels = extract_atom_quantities(raw_datafile_contents)
    velocities = extract_velocities(raw_datafile_contents)

    return {
        "cell": cell,
        "mass": mass,
        "charges": charges,
        "positions": positions,
        "travels": travels,
        "velocities": velocities,
    }


def test_lammpsdata_read(lammpsdata_file_path):
    atoms = ase.io.read(lammpsdata_file_path, format="lammps-data", units="metal")

    expected_values = lammpsdata_file_extracted_groups(lammpsdata_file_path)

    # Check cell was read in correctly
    cell_read_in = atoms.get_cell()
    for cell_vec_read_in, cell_vec_expected in zip(
        cell_read_in, expected_values["cell"]
    ):
        assert cell_vec_read_in == pytest.approx(cell_vec_expected, rel=1e-2)

    # Check masses were read in correctly
    masses_read_in = atoms.get_masses()
    masses_expected = [expected_values["mass"]] * len(expected_values["positions"])
    assert masses_read_in == pytest.approx(masses_expected, rel=1e-2)

    # Check positions were read in correctly
    pos_read_in = atoms.get_positions()
    for pos_read_in, pos_expected in zip(pos_read_in, expected_values["positions"]):
        assert pos_read_in == pytest.approx(pos_expected, rel=1e-2)

    # Check velocities were read in correctly
    vel_read_in = atoms.get_velocities()
    for vel_read_in, vel_expected in zip(vel_read_in, expected_values["velocities"]):
        assert vel_read_in == pytest.approx(vel_expected, rel=1e-2)

    # TODO: Also check charges, travels, molecule id, bonds, and angles
