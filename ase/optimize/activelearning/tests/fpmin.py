from ase.calculators.emt import EMT
from ase.optimize.activelearning.aidmin import AIDMin, SP
from ase.optimize.activelearning.gpfp.fingerprint import OganovFP
from ase.optimize.activelearning.gpfp.kernel import FPKernel
from ase.optimize.activelearning.gpfp.calculator import GPCalculator

from ase.build import bulk
import random

atoms = bulk('Ag', 'fcc')
atoms *= (3,2,1)

atoms.rattle(0.2, seed=random.randint(1,100))
N = len(atoms)
indexes = [0]

for i in indexes:
    atoms[i].symbol='Au'


kernel_params = {'scale':1000, 'weight':1.}
fingerprint = OganovFP
fingerprint_params = {'delta': 0.2, 'N': 200, 'limit': 20}
calculate_uncertainty = False
mask_constraints = False

update_prior_strategy = 'maximum'
kernel = FPKernel()

calculator = GPCalculator(kernel=kernel,
                          noise=0.001,
                          update_prior_strategy=update_prior_strategy,
                          fingerprint=fingerprint,
                          kernel_params=kernel_params,
                          fingerprint_params=fingerprint_params,
                          calculate_uncertainty=calculate_uncertainty,
                          mask_constraints=mask_constraints)

atoms.set_calculator(EMT())

opt = AIDMin(atoms, model_calculator=calculator, optimizer=SP, 
             use_previous_observations=False, surrogate_starting_point='min',
             trainingset=[], print_format='ASE', fit_to='constraints',
             optimizer_kwargs={'fmax': 'scipy default',
                               'method': 'L-BFGS-B'})

opt.run(fmax=0.05)
