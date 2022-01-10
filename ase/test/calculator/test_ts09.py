import pytest
from ase import io
from ase.calculators.vdwcorrection import (vdWTkatchenko09prl,
                                           TS09Polarizability)
from ase.calculators.emt import EMT
from ase.build import bulk, molecule


# fake objects for the test
class FakeHirshfeldPartitioning:
    def __init__(self, calculator):
        self.calculator = calculator

    def initialize(self):
        pass

    def get_effective_volume_ratios(self):
        return [1] * len(self.calculator.atoms)

    def get_calculator(self):
        return self.calculator


class FakeDFTcalculator(EMT):
    def __init__(self, atoms=None):
        self.atoms = atoms
        super().__init__()

    def get_xc_functional(self):
        return 'PBE'


def test_ts09(testdir):
    al = bulk('Al')

    cc = FakeDFTcalculator()
    hp = FakeHirshfeldPartitioning(cc)
    c = vdWTkatchenko09prl(hp, [3])
    al.calc = c
    energy = al.get_potential_energy()

    fname = 'out.traj'
    al.write(fname)

    atoms = io.read(fname)
    assert atoms.get_potential_energy() == pytest.approx(energy)


def test_ts09_polarizability(testdir):
    atoms = molecule('N2')

    cc = FakeDFTcalculator(atoms)
    hp = FakeHirshfeldPartitioning(cc)
    c = vdWTkatchenko09prl(hp, [2, 2])
    atoms.calc = c

    # interface to enable Raman calculations
    pol = TS09Polarizability()
    alpha_cc = pol(atoms)

    # polarizability is a tensor
    assert alpha_cc.shape == (3, 3)

    assert alpha_cc.diagonal() == pytest.approx(0.1523047, .005)
