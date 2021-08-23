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

    def test_atoms_to_etree(self):
        """Test ability to convert ase.Atoms obj to xml element tree."""
        root = ase.io.exciting.atoms_to_etree(
            ase_atoms_obj=self.nitrous_oxide_atoms_obj)
        expected_base_vecs = [
            list(np.divide([0, 0, 1], Bohr)),
            list(np.divide([0, 1, 0], Bohr)),
            list(np.divide([1, 0, 0], Bohr))]
        basis_vecs = root.findall(
            "structure/crystal/basevect")
        basis_vecs = [
            [float(x) for x in basis_vecs[i].text.split()] for i in range(
                len(basis_vecs))]
        print(basis_vecs)

        for i in range(len(expected_base_vecs)):
            for j in range(len(expected_base_vecs[i])):
                self.assertAlmostEqual(
                        basis_vecs[i][j], expected_base_vecs[i][j])

        species = root.findall(
            "structure/species")
        species = [species[i].attrib for i in range(len(species))]
        symbols = [attribute['chemicalSymbol'] for attribute in species]
        species_files = [attribute['speciesfile'] for attribute in species]
        print(symbols)
        expected_chemical_symbols = ['N', 'O']
        expected_species_files = ['N.xml', 'O.xml']
        self.assertListEqual(symbols, expected_chemical_symbols)
        self.assertListEqual(species_files, expected_species_files)


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
