import pytest
from numpy.random import RandomState

from ase.build import molecule
from ase.vibrations import Infrared


class RandomCalculator:
    """Fake Calculator class."""
    def __init__(self):
        self.rng = RandomState(42)

    def get_forces(self, atoms):
        return self.rng.rand(len(atoms), 3)

    def get_dipole_moment(self, atoms):
        return self.rng.rand(3)


def test_folding(testdir):
    """Test that folding is consitent with intensities"""
    
    atoms = molecule('C2H6')
    atoms.calc = RandomCalculator()
    ir = Infrared(atoms)
    ir.run()
    freqs = ir.get_frequencies().real

    for folding in ['Gaussian', 'Lorentzian']:
        x, y = ir.get_spectrum(start=freqs.min() - 100,
                               end=freqs.max() + 100,
                               type=folding,
                               normalize=True)
        assert ir.intensities.sum() == pytest.approx(
            y.sum() * (x[1] - x[0]), 1e-2)
