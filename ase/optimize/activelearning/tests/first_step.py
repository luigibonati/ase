from ase.optimize.activelearning.oldgpmin import OldGPMin
from ase.optimize.activelearning.gp.kernel import BondExponential
from ase.data import covalent_radii, atomic_numbers
from ase.calculators.emt import EMT

from ase.build import bulk
import numpy as np

# We first try with OldGPMin
rc = covalent_radii[atomic_numbers['C']]
atoms = bulk('C', 'fcc', 2*np.sqrt(2)*rc)
atoms *= (2, 2, 2)
atoms.rattle(0.2)
atoms.set_calculator(EMT())

N = len(atoms)

#scale = 0.1*rc

atoms0 = atoms.copy()
# optimizer = OldGPMin(atoms, logfile=None, scale=scale)
# optimizer.run(fmax=0.01, steps=1)

# x0 = atoms0.get_positions().flatten()
# x1 = atoms.get_positions().flatten()

# assert abs(np.sqrt(((x0 - x1) ** 2).sum())-scale) < 0.01

# Lets test it now with the kernel
kernel = BondExponential(3*N)
radii = N * [rc]
interaction = lambda x, y: 1. / (x * y)
kernel.init_metric(radii, interaction, normalize=True)
# atoms = atoms0.copy()
atoms.set_calculator(EMT())

optimizer = OldGPMin(atoms, logfile=None, kernel=kernel, scale=0.1)
optimizer.run(fmax=0.01, steps=1)
x0 = atoms0.get_positions().flatten()
x1 = atoms.get_positions().flatten()
# assert abs(np.sqrt(((x0 - x1) ** 2).sum())-scale) < 0.01

print(np.sqrt(((x0 - x1) ** 2).sum()))
print(rc*0.1)
