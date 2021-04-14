# import inspect
import pytest
import numpy as np
from shutil import copyfile
from ase.io import read


@pytest.fixture
def vasprun(datadir):
    return datadir / 'vasp' / 'vasprun.xml'

def test_positions(vasprun, tmp_path):
    copyfile(vasprun, tmp_path / 'vasprun.xml')
    expected_scaled_positions = np.array([[0.0, 0.0, 0.0],
                                         [0.5,  0.5, 0.5]])

    atoms = read(tmp_path / 'vasprun.xml', index=-1)
    np.testing.assert_allclose(atoms.get_scaled_positions(),
                               expected_scaled_positions)
    np.testing.assert_allclose(atoms.positions,
                               expected_scaled_positions @
                               atoms.cell.complete())
