from ase.test.datadriven.main import test_singlepoint
from ase.calculators.datadriven import new_emt
from ase.build import bulk

atoms = bulk('Au') * (2, 2, 2)
atoms.rattle(stdev=0.2)
atoms.calc = new_emt()

test_singlepoint('emt', atoms)
