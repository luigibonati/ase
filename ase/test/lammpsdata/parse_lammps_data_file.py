"""
Routines for manually parsing a lammps data file.  This is a simplified
recreation of ase.io.lammpsdata's read functionality that we use for
testing so as to attempt to provide an independent verification.  It is
fairly ad hoc and was designed specifically to work for the
'lammpsdata_input.data' file under ase.test.testdata.  In particular,
the following assumptions are made about the lammps data file:

  - Only a single atomic species is present

  - The cell is orthogonal (xy, xz, yz tilt factors are
    ignored even if they exist)

  - The sections are given in the following order:
        Masses -> Atoms -> Velocities -> Bonds -> Angles
"""
import re

import numpy as np
from ase.calculators.lammps import convert

def extract_cell(raw_datafile_contents):
    """
    NOTE: Assumes an orthogonal cell (xy, xz, yz tilt factors are
    ignored even if they exist)
    """
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
    NOTE: Assumes that only a single atomic species is present and that
    Masses section is followed immediately by Atoms section
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
    """
    NOTE: Assumes Atoms section is immediately followed by Velocities
    section
    """
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
    NOTE: Assumes metal units are used in data file and that Velocities
    section is followed immediately by Bonds section
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


def lammpsdata_file_extracted_sections(lammpsdata_file_path):
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
