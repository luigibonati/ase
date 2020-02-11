from ase.optimize.activelearning.gpfp.fingerprint import RadialAngularFP
from ase.optimize.activelearning.gpfp.calculator import GPCalculator
from ase.optimize.activelearning.gpfp.kernel import FPKernel
import numpy as np
import copy

from ase.build import fcc100
from ase.calculators.emt import EMT


def set_structure(atoms, seed=0):
    """
    Create a copy and rattle.
    """
    newatoms = copy.deepcopy(atoms)
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
fp = RadialAngularFP
kernel = FPKernel()
params = {'weight': 1.0, 'scale': 50, 'delta': 0.2}

calc = GPCalculator(train_images=train_images, noise=1e-3,
                    kernel=kernel, params=params,
                    update_prior_strategy='maximum',
                    params_to_update={'weight': (0.1, np.inf),
                                      'scale': (0.01, np.inf)},
                    batch_size=1,
                    print_format='ASE',
                    fingerprint=fp,
                    mask_constraints=False)

slab.set_calculator(calc)

print('GP energy: {}eV'.format(slab.get_potential_energy()))
