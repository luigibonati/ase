import pytest

from ase.calculators.orca import ORCA
from ase.atoms import Atoms

@pytest.fixture
def water():
    return Atoms('OHH', positions=[(0, 0, 0), (1, 0, 0), (0, 1, 0)])

def test_orca(water):
    from ase.optimize import BFGS

    water.calc = ORCA(label='water',
                      orcasimpleinput='BLYP def2-SVP')

    with BFGS(water) as opt:
        opt.run(fmax=0.05)

    final_energy = water.get_potential_energy()
    print(final_energy)

    assert abs(final_energy + 2077.24420) < 1.0

