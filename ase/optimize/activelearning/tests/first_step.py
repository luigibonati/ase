from ase.optimize.activelearning.oldgpmin import OldGPMin
from ase.data import covalent_radii, atomic_numbers
from ase.calculators.emt import EMT

from ase.build import bulk
import numpy as np

rc = covalent_radii[atomic_numbers['C']]
atoms = bulk('C', 'fcc', 2*np.sqrt(2)*rc)
atoms *= (2, 2, 2)
atoms.rattle(0.2)
atoms.set_calculator(EMT())

scale = 0.1*rc

atoms0 = atoms.copy()
optimizer = OldGPMin(atoms, logfile='-', scale=scale)
optimizer.run(fmax=0.01, steps=1)

x0 = atoms0.get_positions().flatten()
x1 = atoms.get_positions().flatten()

print(np.sqrt(((x0 - x1) ** 2).sum()))
print(rc * scale)
