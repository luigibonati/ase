from ase.cluster import wulff_construction
from ase.constraints import FixAtoms

from ase.calculators.emt import EMT
from ase.optimize import *
from ase.optimize.bayesian.lgpmin import LGPMin, GPModel
import time

""" 
    Structure relaxation of Pt heptamer island on Pt(111).
    Benchmark using LGPMin, GPMin, LBFGS and FIRE. 
"""

# 1. Build Atoms Object.
###############################################################################

# Setup calculator:
calc = EMT()

# 1.1. Set up structure:

surfaces = [(1, 0, 0), (1, 1, 0), (1, 1, 1)]
esurf = [1.0, 1.1, 0.9]
lc = 4.0
size = 200
atoms = wulff_construction('Au', surfaces, esurf,
                           size, 'fcc',
                           rounding='above', latticeconstant=lc)
atoms.center(vacuum=5.0)
atoms.rattle(stdev=0.10, seed=0)

c = FixAtoms(indices=[0])
atoms.set_constraint(c)


# 2. Benchmark.
###############################################################################

# 2.A. Optimize structure using LGPMin.
initial_lgpmin = atoms.copy()
initial_lgpmin.set_calculator(calc)
gp_model = GPModel(scale=0.4, max_training_data=10,
                   max_train_data_strategy='lowest_energy')
lgpmin_opt = LGPMin(initial_lgpmin, model=gp_model)
start = time.time()
lgpmin_opt.run(fmax=0.01, restart=False)
end = time.time()
print('Time LGPMin: ', end-start)


# 2.A. Optimize structure using GPMin.
initial_gpmin = atoms.copy()
initial_gpmin.set_calculator(calc)
gpmin_opt = GPMin(initial_gpmin)
start = time.time()
gpmin_opt.run(fmax=0.01)
end = time.time()
print('Time GPMin: ', end-start)