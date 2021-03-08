import os
import pytest
from numpy.random import RandomState

from ase.build import molecule
from ase.vibrations import Vibrations
from ase.vibrations import Infrared


@pytest.fixture
def atoms():
    """Define atoms and do the IR calculation"""
    
    class RandomCalculator():
        """Fake Calculator class.

        """
        def __init__(self):
            self.rng = RandomState(42)

        def get_forces(self, atoms):
            return self.rng.rand(len(atoms), 3)

        def get_dipole_moment(self, atoms):
            return self.rng.rand(3)
    
    atoms = molecule('C2H6')
    ir = Infrared(atoms)
    ir.calc = RandomCalculator()
    ir.run()

    return atoms

        
def test_combine(atoms):

    ir = Infrared(atoms)

    freqs = ir.get_frequencies()
    ints = ir.intensities
    assert ir.combine() == 49

    ir = Infrared(atoms)
    assert (freqs == ir.get_frequencies()).all()
    assert (ints == ir.intensities).all()

    vib = Vibrations(atoms, name='ir')
    assert (freqs == vib.get_frequencies()).all()

    # Read the data from other working directory
    dirname = os.path.basename(os.getcwd())
    os.chdir('..')  # Change working directory
    ir = Infrared(atoms, name=os.path.join(dirname, 'ir'))
    assert (freqs == ir.get_frequencies()).all()
    os.chdir(dirname)

    ir = Infrared(atoms)
    assert ir.split() == 1
    assert (freqs == ir.get_frequencies()).all()
    assert (ints == ir.intensities).all()

    vib = Vibrations(atoms, name='ir')
    assert (freqs == vib.get_frequencies()).all()

    assert ir.clean() == 49


def test_folding(atoms):
    """Test that folding is consitent with intensities"""
    
    ir = Infrared(atoms)
    freqs = ir.get_frequencies().real

    for folding in ['Gaussian', 'Lorentzian']:
        x, y = ir.get_spectrum(start=freqs.min() - 100,
                               end=freqs.max() + 100,
                               type=folding,
                               normalize=True)
        assert ir.intensities.sum() == pytest.approx(
            y.sum() * (x[1] - x[0]), 1e-2)
