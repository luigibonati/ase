from ase.io import read
from gpaw import GPAW
from ase.optimize.activelearning.lgpmin import LGPMin
from ase.calculators.gp.calculator import GPCalculator

atoms = read('molecule.xyz')
atoms.center(vacuum=4.5)
calc = GPAW(mode='lcao', basis='sz(dzp)', h=0.3,
            txt='gpaw_output.txt',
            convergence=dict(density=1e-2))
atoms.calc = calc

gp_model = GPCalculator(scale=0.4, max_train_data=20,
                        max_train_data_strategy='last_observations')
opt = LGPMin(atoms, model_calculator=gp_model)

opt.run(fmax=0.01, restart=False)
