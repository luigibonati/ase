"""
Create an atoms object and write it to a lammps data file
"""
from io import StringIO

import pytest
import ase.io

from .parse_lammps_data_file import lammpsdata_file_extracted_sections

# Relative tolerance for comparing floats with pytest.approx
REL_TOL = 1e-2


def test_lammpsdata_write(atoms):
    # Write atoms object to lammpsdata file-like object
    lammpsdata_buf = StringIO()
    ase.io.write(
        lammpsdata_buf, atoms, format="lammps-data", atom_style="full", velocities=True
    )

    # Now read the output back, parse it, and compare to the original atoms
    # object attributes
    written_values = lammpsdata_file_extracted_sections(lammpsdata_buf)

    # Check cell was written correctly
    cell_written = written_values["cell"]
    cell_expected = atoms.get_cell()
    for cell_vec_written, cell_vec_expected in zip(cell_written, cell_expected):
        assert cell_vec_written == pytest.approx(cell_vec_expected, rel=REL_TOL)

    # Check that positions were written correctly
    positions_written = written_values["positions"]
    positions_expected = atoms.get_positions(wrap=True)
    for pos_written, pos_expected in zip(positions_written, positions_expected):
        assert pos_written == pytest.approx(pos_expected, rel=REL_TOL)

    # Check velocities were read in correctly
    velocities_written = written_values["velocities"]
    velocities_expected = atoms.get_velocities()
    for vel_written, vel_expected in zip(velocities_written, velocities_expected):
        assert vel_written == pytest.approx(vel_expected, rel=REL_TOL)
