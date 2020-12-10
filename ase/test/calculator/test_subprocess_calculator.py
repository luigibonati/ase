import pytest
from ase.calculators.subprocesscalculator import NamedPackedCalculator
from ase.calculators.emt import EMT
from ase.build import bulk
from ase.optimize import BFGS
from ase.constraints import ExpCellFilter


def get_fmax(forces):
    return max((forces**2).sum(axis=1)**0.5)


@pytest.fixture
def atoms():
    atoms = bulk('Au') * (2, 2, 2)
    atoms.rattle(stdev=0.05, seed=2)
    return atoms


def assert_equal_with_emt(atoms):
    atoms1 = atoms.copy()
    atoms1.calc = EMT()

    assert (atoms.get_potential_energy()
            == pytest.approx(atoms1.get_potential_energy()))
    assert atoms.get_forces() == pytest.approx(atoms1.get_forces())
    assert atoms.get_stress() == pytest.approx(atoms1.get_stress())


def test_subprocess_calculator(atoms):
    pack = NamedPackedCalculator('emt')

    with pack.calculator() as atoms.calc:
        assert_equal_with_emt(atoms)
        atoms.rattle(stdev=0.1)
        atoms.cell[1, 2] += 0.1
        atoms.pbc[0] = False
        assert_equal_with_emt(atoms)

    # As before, but the results are supposedly cached:
    assert_equal_with_emt(atoms)

def test_subprocess_calculator_optimize(atoms):
    pack = NamedPackedCalculator('emt')
    opt = BFGS(atoms)

    fmax = 0.05
    with pack.calculator() as atoms.calc:
        fmax_start = get_fmax(atoms.get_forces())
        assert fmax_start > fmax
        opt.run(fmax=fmax)
        atoms.get_stress()

    fmax_now = get_fmax(atoms.get_forces())
    assert fmax_now < fmax
    assert_equal_with_emt(atoms)
