from ase.test.datadriven.main import test_singlepoint
from ase.calculators.datadriven import new_espresso
from ase.build import bulk

atoms = bulk('Si')
atoms.rattle(stdev=0.1)


PSEUDO = {'Si': 'Si.rel-pbe-rrkj.UPF'}

calc = new_espresso(kpts=[2, 2, 2], ecutwfc=20.0,
                    pseudopotentials=PSEUDO,
                    tprnfor=True)
atoms.calc = calc
print(calc)

test_singlepoint('espresso', atoms)
#f = atoms.get_forces()
#print('f', f)
#e = atoms.get_potential_energy()
#print('e', e)
