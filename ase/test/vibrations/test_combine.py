from numpy.random import RandomState

from ase.build import molecule
from ase.vibrations import Vibrations, Infrared
from ase.utils import workdir


class RandomCalculator:
    """Fake Calculator class."""
    def __init__(self):
        self.rng = RandomState(42)

    def get_forces(self, atoms):
        return self.rng.rand(len(atoms), 3)

    def get_dipole_moment(self, atoms):
        return self.rng.rand(3)


def test_combine(testdir):
    dirname = 'subdir'
    vibname = 'ir'
    with workdir(dirname, mkdir=True):
        atoms = molecule('C2H6')
        ir = Infrared(atoms)
        assert ir.name == vibname
        ir.calc = RandomCalculator()
        ir.run()
        freqs = ir.get_frequencies()
        ints = ir.intensities
        assert ir.combine() == 49

        ir = Infrared(atoms)
        assert (freqs == ir.get_frequencies()).all()
        assert (ints == ir.intensities).all()

        vib = Vibrations(atoms, name=vibname)
        assert (freqs == vib.get_frequencies()).all()

        # Read the data from other working directory
        with workdir('..'):
            ir = Infrared(atoms, name=f'{dirname}/{vibname}')
            assert (freqs == ir.get_frequencies()).all()

        ir = Infrared(atoms)
        assert ir.split() == 1
        assert (freqs == ir.get_frequencies()).all()
        assert (ints == ir.intensities).all()

        vib = Vibrations(atoms, name=vibname)
        assert (freqs == vib.get_frequencies()).all()

        assert ir.clean() == 49
