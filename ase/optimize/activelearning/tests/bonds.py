import numpy as np
import copy

from ase.build import bulk 
from ase.calculators.emt import EMT

from ase.optimize.activelearning.gp.calculator import GPCalculator
from ase.optimize.activelearning.gp.kernel import BondExponential

from ase.data import covalent_radii


def get_structure(atoms):
    """
    Create a copy and rattle.
    """
    newatoms = copy.deepcopy(atoms)
    newatoms.rattle(0.1, seed=np.random.randint(10000))
    return newatoms

# Atoms 
atoms = bulk('C', 'diamond')
atoms.set_calculator(EMT())
true_e = atoms.get_potential_energy(force_consistent=True)

# Training set
train_images = []
for i in range(5):
    atoms2 = get_structure(atoms)
    train_images.append(atoms2)

# Kernel
radii = [covalent_radii[n] for n in atoms.get_atomic_numbers()]
kernel = BondExponential(3*len(atoms))
kernel.init_metric(radii)

kernel_params = {'weight':1.0, 'scale': 0.7}

calc = GPCalculator(train_images=train_images, noise=1e-3,
                    kernel=kernel, kernel_params=kernel_params,
                    update_prior_strategy='maximum',
                    mask_constraints=False)

atoms.set_calculator(calc)
print(true_e, atoms.get_potential_energy())
