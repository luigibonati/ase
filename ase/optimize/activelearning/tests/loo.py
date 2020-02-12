from ase.optimize.activelearning.gpfp.fingerprint import OganovFP
from ase.optimize.activelearning.gpfp.calculator import GPCalculator
from ase.optimize.activelearning.gpfp.kernel import FPKernel
import numpy as np
import copy
import time

from ase.build import fcc100
from ase.calculators.emt import EMT


def set_structure(atoms, seed=0):
    """
    Create a copy and rattle.
    """
    newatoms = atoms.copy()
    newatoms.set_calculator(EMT())
    newatoms.rattle(0.1, seed=seed)
    return newatoms


# Create slab
slab = fcc100('Ag', size=(2, 2, 2))
slab[-1].symbol = 'Au'
slab[-2].symbol = 'Au'
slab.rattle(0.5)
slab.center(vacuum=4.0)
slab.pbc = False
slab.set_calculator(EMT())

# Create Training Set
train_images = []
for i in range(5):
    # Alter structure and get fingerprint
    slab2 = set_structure(slab, seed=i)
    train_images.append(slab2)

print('EMT energy: {}eV'.format(slab.get_potential_energy()))

# Initialize fingerprint
fp = OganovFP
kernel = FPKernel()
params = {'weight': 1, 'scale': 1, 'delta': 0.2}
noise = 1e-3
calc = GPCalculator(train_images=train_images, noise=noise,
                    kernel=kernel, params=params,
                    update_prior_strategy='maximum',
                    params_to_update={'scale': (0.01, np.inf),},
                    batch_size=1,
                    print_format='ASE',
                    fingerprint=fp,
                    mask_constraints=False)

slab.set_calculator(calc)
print(slab.get_potential_energy())

predes = []
reales = []
for index in range(5):
    t0 = time.time()
    calc.calculate_LOO(index)
    predes.append(calc.results['looenergy'])
    reales.append(calc.train_y[index][0])
    print("Leave-one-out prediction: {:.04f} Correct energy: {:.04f} Time: {:.04f} sec".format(predes[-1], reales[-1], (time.time()-t0)))
rmse = np.sqrt(((np.array(predes) - np.array(reales)) ** 2).mean())
print ('RMSE: {:.04f} eV'.format(rmse))

slab.set_calculator(calc)
print(slab.get_potential_energy())

# Repeat to test:
predes = []
reales = []
for index in range(5):
    t0 = time.time()
    calc.calculate_LOO(index)
    predes.append(calc.results['looenergy'])
    reales.append(calc.train_y[index][0])
    print("Leave-one-out prediction: {:.04f} Correct energy: {:.04f} Time: {:.04f} sec".format(predes[-1], reales[-1], (time.time()-t0)))
rmse = np.sqrt(((np.array(predes) - np.array(reales)) ** 2).mean())
print ('RMSE: {:.04f} eV'.format(rmse))

slab.set_calculator(calc)
print(slab.get_potential_energy())
