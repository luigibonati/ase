from ase.build import bulk
from ase.optimize.activelearning.gpfp.calculator import copy_image
from ase.calculators.emt import EMT

atoms = bulk('Cu', 'fcc')
atoms.set_calculator(EMT())

atoms0 = copy_image(atoms)
print(atoms0.calc.__class__.__name__)
print(atoms0.get_potential_energy(force_consistent=False))
print(atoms0.get_forces())
