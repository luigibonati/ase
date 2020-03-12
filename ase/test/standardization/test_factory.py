from ase.build import molecule
from ase.units import Rydberg
from ase.calculators.factory import get_calculator
import pytest


calcs = {
    'gamess-us': dict(),
    'espresso': dict(pseudopotentials={'C': 'C.pz-kjpaw.UPF',
                                       'H': 'H.pz-kjpaw.UPF'},
                     ecutwfc=300 / Rydberg),
    'abinit': dict(ecut=300, chksymbreak=0, toldfe=1e-4),
}


def _calculate(calcname, species, parameters):
    atoms = molecule(species)
    atoms.center(vacuum=3.5)
    atoms.calc = get_calculator(calcname, **parameters)
    return atoms.get_potential_energy()


@pytest.mark.parametrize('calc', calcs)
def test_ch4(calc):
    e_ch4 = _calculate(calc, 'CH4', calcs.get(calc))
    e_c2h2 = _calculate(calc, 'C2H2', calcs.get(calc))
    e_h2 = _calculate(calc, 'H2', calcs.get(calc))
    energy = e_ch4 - 0.5 * e_c2h2 - 1.5 * e_h2
    print(energy)
    assert energy == pytest.approx(-2.8, abs=0.3)
