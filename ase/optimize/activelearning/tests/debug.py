from ase.optimize.activelearning.gpfp.calculator import GPCalculator
from ase.optimize.activelearning.gpfp.kernel import FPKernel
from ase.optimize.activelearning.gpfp.fingerprint import OganovFP

from ase.build import bulk
from ase.calculators.emt import EMT
import random
import numpy as np
from copy import copy

# Make atoms object
atoms = bulk('Ag', 'fcc')
atoms *= (3,2,1)
atoms.rattle(0.2, seed=0)

random.seed(0)

N = len(atoms)
indexes = random.sample(range(N), N//2)
for i in indexes:
    atoms[i].symbol='Au'

# Create a 1-point training set
ml_atoms = atoms.copy()
atoms.set_calculator(EMT())

e = atoms.get_potential_energy()
f = atoms.get_forces()

# Define calculator
x = OganovFP(limit=20.0, delta=0.2, N=200)
x.set_atoms(ml_atoms)

kernel = FPKernel()
kernel.set_params({'scale':1.1, 'weight':4.5})

noise = 0.0005/4.5


x_train = OganovFP(limit=20.0, delta=0.2, N=200)
x_train.set_atoms(atoms)

X = np.array([copy(x_train)])
Y = np.insert(-f.flatten(), 0, e).reshape(1, -1)

K = kernel.kernel_matrix(X)

print(K)
