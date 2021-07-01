import pytest
from ase import io
from ase.calculators.vdwcorrection import vdWTkatchenko09prl
from ase.calculators.emt import EMT
from ase.build import bulk, molecule


def test_ts09(testdir):

    # fake objects for the test
    class FakeHirshfeldPartitioning:
        def __init__(self, calculator):
            self.calculator = calculator

        def initialize(self):
            pass

        def get_effective_volume_ratios(self):
            return [1]

        def get_calculator(self):
            return self.calculator

    class FakeDFTcalculator(EMT):
        def get_xc_functional(self):
            return 'PBE'

    a = 4.05  # Angstrom lattice spacing
    al = bulk('Al', 'fcc', a=a)

    cc = FakeDFTcalculator()
    hp = FakeHirshfeldPartitioning(cc)
    c = vdWTkatchenko09prl(hp, [3])
    al.calc = c
    al.get_potential_energy()

    fname = 'out.traj'
    al.write(fname)

    # check that the output exists
    io.read(fname)
    # maybe assert something about what we just read?

    p = io.read(fname).calc.parameters
    p['calculator']
    p['xc']
    p['uncorrected_energy']


def test_ts09_polarizability(testdir):

    # fake objects for the test
    class FakeHirshfeldPartitioning:
        def __init__(self, calculator):
            self.calculator = calculator

        def initialize(self):
            pass

        def get_effective_volume_ratios(self):
            return [1.] * len(atoms)

        def get_calculator(self):
            return self.calculator

    class FakeDFTcalculator(EMT):
        def get_xc_functional(self):
            return 'PBE'

        def get_atoms(self):
            return atoms

    atoms = molecule('N2')

    cc = FakeDFTcalculator()
    hp = FakeHirshfeldPartitioning(cc)
    c = vdWTkatchenko09prl(hp, [3])

    atoms.calc = c
    alpha = c.get_polarizability()
    assert alpha == pytest.approx(14.8, .5)
