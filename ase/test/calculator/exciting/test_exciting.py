import pytest
from ase.build import bulk


import ase.io
import ase.calculators.exciting as exciting
import unittest

@pytest.mark.calculator_lite
@pytest.mark.calculator('exciting')
def test_exciting_bulk(factory):
    atoms = bulk('Si')
    atoms.calc = factory.calc()
    energy = atoms.get_potential_energy()
    print(energy)


class TestExciting(unittest.TestCase):
    def test_exciting_constructor(self):
        """Test write an input for exciting."""
        a = Atoms('N3O',
              [(0, 0, 0), (1, 0, 0), (0, 0, 1), (0.5, 0.5, 0.5)],
              pbc=True)
        exc_calc = exciting.Exciting(
            dir='ase/test/calculator/exciting',
            kpts=(3, 3, 3),
            species_path='/None',
            exciting_binary='/fshome/chm/git/exciting/bin/excitingser',
            maxscl=3)
        self.assertEqual(
                exc_calc.groundstate_attributes['ngridk'],
                '3 3 3')
        self.assertEqual(exc_calc.dir, 'ase/test/calculator/exciting')
        self.assertEqual(exc_calc.species_path, '/None')
        self.assertEqual(exc_calc.binary_path)



