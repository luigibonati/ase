from ase.test.datadriven.main import test_singlepoint
from ase.calculators.datadriven import new_emt

test_singlepoint('emt', new_emt())
