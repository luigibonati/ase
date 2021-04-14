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
    expected_scaled_positions = np.array([[0.0, 0.0, 0.0],
                                         [0.5,  0.5, 0.5]])

    atoms = read(tmp_path / 'vasprun.xml', index=-1)

    # chek number of atoms
    assert len(atoms)==2

    # check scaled_positions
    np.testing.assert_allclose(atoms.get_scaled_positions(),
                               expected_scaled_positions)

    expected_cell = np.array([[3.16, 0.0, 0.0],
                              [0.0, 3.16, 0.0],
                              [0.0, 0.0, 3.16]])

    # check cell
    np.testing.assert_allclose(atoms.cell, expected_cell)

    # check real positions
    np.testing.assert_allclose(atoms.positions,
                               expected_scaled_positions @
                               atoms.cell.complete())
