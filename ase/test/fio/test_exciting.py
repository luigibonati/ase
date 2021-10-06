"""Test file for exciting file input and output methods."""
import pytest
import tempfile  # Used to create temporary directories for tests.
import unittest

import ase
import ase.io.exciting
from ase.io import read, write
from ase.units import Bohr
import numpy as np


class TestExciting(unittest.TestCase):

    def setUp(self):
        self.nitrous_oxide_atoms_obj = ase.Atoms(
            'N3O',
            positions=[
                (0, 0, 0), (1, 0, 0),
                (0, 0, 1), (0.5, 0.5, 0.5)],
            cell=[[0, 0, 1], [0, 1, 0], [1, 0, 0]],
            pbc=True)
        self.test_folder_name = tempfile.mkdtemp()


def test_exciting_io():
    """Old test that should be depracated."""
    atoms = ase.Atoms('N3O',
                  cell=[3, 4, 5],
                  positions=[(0, 0, 0), (1, 0, 0),
                             (0, 0, 1), (0.5, 0.5, 0.5)],
                  pbc=True)

    write('input.xml', atoms)
    atoms2 = read('input.xml')

    assert all(atoms.symbols == atoms2.symbols)
    assert atoms.cell[:] == pytest.approx(atoms2.cell[:])
    assert atoms.positions == pytest.approx(atoms2.positions)
