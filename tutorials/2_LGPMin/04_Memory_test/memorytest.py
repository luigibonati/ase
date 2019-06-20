from ase.cluster import wulff_construction
from ase.constraints import FixAtoms

from ase.calculators.emt import EMT
from ase.optimize import *
from ase.optimize.activelearning.lgpmin import LGPMin
from ase.calculators.gp.calculator import GPCalculator
import time

""" 
    Structure relaxation of n Au cluster.
    Benchmark using LGPMin, GPMin memory consumption and function evaluations. 
"""

list_cluster_sizes = [10, 20, 30, 40, 50, 60, 80, 100, 120, 150, 200]

for cluster_size in list_cluster_sizes:  # Loop over cluster sizes.

    for seed in range(10):   # Loop over different rattles.
        # 1.1. Set up structure:
        surfaces = [(1, 0, 0), (1, 1, 0), (1, 1, 1)]
        esurf = [1.0, 1.1, 0.9]
        lc = 4.0
        atoms = wulff_construction('Au', surfaces, esurf,
                                   cluster_size, 'fcc',
                                   rounding='above', latticeconstant=lc)
        atoms.center(vacuum=5.0)
        atoms.rattle(stdev=0.10, seed=0)
        c = FixAtoms(indices=[0])
        atoms.set_constraint(c)
        atoms.rattle(stdev=0.1, seed=seed)

        # 1.2. Setup calculator:
        calc = EMT()

        # 2.A. Optimize structure using GPMin.
        initial_gpmin = atoms.copy()
        initial_gpmin.set_calculator(calc)
        gpmin_opt = GPMin(initial_gpmin)
        start = time.time()
        gpmin_opt.run(fmax=0.01)
        end = time.time()
        elapsed_time_gpmin = end - start
        size_cov_gpmin = ((len(atoms) * 3) * gpmin_opt.function_calls +
                          gpmin_opt.function_calls)**2
        print('\n Time GPMin: ', elapsed_time_gpmin)
        print('Size covariance LGPMin: ', size_cov_gpmin)

        # 2.B. Optimize structure using LGPMin.
        initial_lgpmin = atoms.copy()
        initial_lgpmin.set_calculator(calc)
        gp_model = GPCalculator(scale=0.4, max_train_data=20,
                                max_train_data_strategy='last_observations')
        lgpmin_opt = LGPMin(initial_lgpmin, model_calculator=gp_model)
        start = time.time()
        lgpmin_opt.run(fmax=0.01, restart=False)
        end = time.time()
        elapsed_time_lgpmin = end - start
        max_data = lgpmin_opt.model_calculator.max_data
        if lgpmin_opt.model_calculator.max_data_strategy == 'nearest_train':
            max_data += 2

        size_cov_lgpmin = ((len(atoms) * 3) * max_data + max_data)**2
        print('\n Time LGPMin: ', elapsed_time_lgpmin)
        print('Size covariance LGPMin: ', size_cov_lgpmin)
