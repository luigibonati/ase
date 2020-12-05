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
"""
import io
import re
import pathlib

import numpy as np
from ase.calculators.lammps import convert


def split_contents_by_section(raw_datafile_contents):
    return re.split(r"^([A-Za-z]+\s*)$\n", raw_datafile_contents, flags=re.MULTILINE)


def extract_cell(raw_datafile_contents):
    """
    NOTE: Assumes an orthogonal cell (xy, xz, yz tilt factors are
    ignored even if they exist)
    """
    RE_CELL = re.compile(
        r"""
            ([0-9e+.-]+)\s+([0-9e+.-]+)\s+xlo\s+xhi\n
            ([0-9e+.-]+)\s+([0-9e+.-]+)\s+ylo\s+yhi\n
            ([0-9e+.-]+)\s+([0-9e+.-]+)\s+zlo\s+zhi\n
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
    contents_split_by_section = split_contents_by_section(raw_datafile_contents)

    masses_block = None

    # Grab all atoms lines
    for ind, block in enumerate(contents_split_by_section):
        if block.startswith("Masses"):
            masses_block = contents_split_by_section[ind + 1].strip()
            break

    if masses_block is None:
        return None
    else:
        mass = re.match(r"\s*[0-9]+\s+([0-9e+.-]+)", masses_block).group(1)
        return float(mass)


def extract_atom_quantities(raw_datafile_contents):
    contents_split_by_section = split_contents_by_section(raw_datafile_contents)

    # Grab all atoms lines
    for ind, block in enumerate(contents_split_by_section):
        if block.startswith("Atoms"):
            positions_block = contents_split_by_section[ind + 1].strip()
            break

    # Now parse each individual atoms line for quantities
    charges = []
    positions = []
    travels = []

    for atom_line in positions_block.splitlines():
        RE_ATOM_LINE = re.compile(
            r"\s*[0-9]+\s+[0-9]+\s+[0-9]+\s+([0-9e+.-]+)\s+"
            r"([0-9e+.-]+)\s+([0-9e+.-]+)\s+([0-9e+.-]+)\s?"
            r"([0-9-]+)?\s?([0-9-]+)?\s?([0-9-]+)?"
        )
        q, x, y, z, *travel = RE_ATOM_LINE.match(atom_line).groups()
        charges.append(float(q))
        positions.append(list(map(float, [x, y, z])))
        if None not in travel:
            travels.append(list(map(int, travel)))
        else:
            travels.append(None)

    return charges, positions, travels


def extract_velocities(raw_datafile_contents):
    """
    NOTE: Assumes metal units are used in data file
    """
    contents_split_by_section = split_contents_by_section(raw_datafile_contents)

    # Grab all velocities lines
    for ind, block in enumerate(contents_split_by_section):
        if block.startswith("Velocities"):
            velocities_block = contents_split_by_section[ind + 1].strip()
            break

    # Now parse each individual line for velocity
    velocities = []
    for velocities_line in velocities_block.splitlines():
        RE_VELOCITY = re.compile(
            r"\s*[0-9]+\s+([0-9e+.-]+)\s+([0-9e+.-]+)" r"\s+([0-9e+.-]+)"
        )
        v = RE_VELOCITY.match(velocities_line).groups()
        velocities.append(list(map(float, v)))

    # Convert to ASE's velocity units (uses unusual unit of time)
    velocities = convert(np.asarray(velocities), "velocity", "metal", "ASE")

    return velocities


def lammpsdata_file_extracted_sections(lammpsdata):
    """
    Manually read a lammpsdata file and grep for the different
    quantities we want to check.  Accepts either a string indicating the name
    of the file, a pathlib.Path object indicating the location of the file, a
    StringIO object containing the file contents, or a file object
    """
    if isinstance(lammpsdata, str) or isinstance(lammpsdata, pathlib.Path):
        with open(lammpsdata) as f:
            raw_datafile_contents = f.read()

    elif isinstance(lammpsdata, io.StringIO):
        raw_datafile_contents = lammpsdata.getvalue()

    elif isinstance(lammpsdata, io.TextIOBase):
        raw_datafile_contents = lammpsdata.read()

    else:
        raise ValueError(
            "Lammps data file content inputted in unsupported "
            "object type {type(lammpsdata)}"
        )

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
