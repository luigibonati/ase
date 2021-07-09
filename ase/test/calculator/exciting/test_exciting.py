"""Test file for exciting ASE calculator."""
from parameterized import parameterized
from collections.abc import Sequence
from typing import Tuple
import unittest
import tempfile  # Used to create temporary directories for tests.

from ase.build import bulk
import ase.calculators.exciting as exciting


# @pytest.mark.calculator_lite
# @pytest.mark.calculator('exciting')
# def test_exciting_bulk(factory):
#     """System level test. Ensure that at least the call doesn't fail."""
#     atoms = bulk('Si')
#     atoms.calc = factory.calc()
#     energy = atoms.get_potential_energy()
#     print(energy)


class TestExciting(unittest.TestCase):
    """Test class for all exciting unit tests."""

    def setUp(self):
        """Code to use for all tests at runtime."""
        self.test_folder_name = tempfile.mkdtemp()

    @parameterized.expand([[
        (3, 3, 3), '/fshome/chm/git/exciting/bin/excitingser',
        '3 3 3', '/fshome/chm/git/exciting/bin/excitingser'],
        [(1, 2, 3), '/foo/bar',
         '1 2 3', '/foo/bar']
    ])
    def test_exciting_constructor(self, kpts: Tuple[int], exciting_binary: str, expected_kpts: str,
                                  expected_exciting_binary: str):
        """Test write an input for exciting."""
        calc_dir = 'ase/test/calculator/exciting'
        exciting_calc = exciting.Exciting(
            dir=calc_dir,
            kpts=kpts,
            species_path=self.test_folder_name,
            exciting_binary=exciting_binary,
            maxscl=3)
        # groundstate attribute ngridk returns the calculator's kpts
        self.assertEqual(
            exciting_calc.groundstate_attributes['ngridk'], expected_kpts)
        self.assertEqual(exciting_calc.dir, calc_dir)
        self.assertEqual(exciting_calc.species_path, self.test_folder_name)
        self.assertEqual(exciting_calc.exciting_binary, expected_exciting_binary)
        # Should be set to False at initialization.
        self.assertFalse(exciting_calc.converged)
        # Should be false by default unless arg is passed to constructor.
        self.assertFalse(exciting_calc.autormt)
        # Should be true by default unless arg is passed to constructor.
        self.assertTrue(exciting_calc.tshift)

    def test_write(self):
        """Test the write method"""
        pass
