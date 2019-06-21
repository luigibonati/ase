from ase.cluster import wulff_construction
from ase.constraints import FixAtoms

from ase.calculators.emt import EMT
from ase.optimize import *
from ase.optimize.activelearning.lgpmin import LGPMin
from ase.calculators.gp.calculator import GPCalculator
import time
import pandas as pd

""" 
    Structure relaxation of n Au cluster.
    Benchmark using LGPMin, GPMin memory consumption and function evaluations. 
"""


"""TABLE
cluster_size -- seed -- algorithm -- max_data -- mode_max_data -- energy 
-- feval -- elapsed_time -- covariance_size
"""
results = {'cluster_size': [], 'seed': [], 'algorithm': [], 'max_data': [],
            'mode_max_data': [], 'energy': [], 'feval': [],
            'elapsed_time': [], 'covariance_size': []}

list_cluster_sizes = [10, 25, 50, 75, 100, 150, 200, 250]
list_max_data = [10, 15, 20, 25, 30, 40, 50]
max_data_mode = ['last_observations', 'lowest_energy', 'nearest_observations']
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
        print('Feval GPMin:', gpmin_opt.function_calls)

        results['cluster_size'].append(cluster_size)
        results['seed'].append(seed)
        results['algorithm'].append('GPMin')
        results['max_data'].append('NA')
        results['mode_max_data'].append('NA')
        results['energy'].append(initial_gpmin.get_potential_energy())
        results['feval'].append(gpmin_opt.function_calls)
        results['elapsed_time'].append(elapsed_time_gpmin)
        results['covariance_size'].append(size_cov_gpmin)

        # 2.B. Optimize structure using LGPMin.
        for max_data in list_max_data:
            for data_mode in max_data_mode:
                initial_lgpmin = atoms.copy()
                initial_lgpmin.set_calculator(calc)
                gp_model = GPCalculator(scale=0.4, max_train_data=max_data,
                                        max_train_data_strategy=data_mode)
                lgpmin_opt = LGPMin(initial_lgpmin, restart=False,
                                    model_calculator=gp_model)
                start = time.time()
                lgpmin_opt.run(fmax=0.01)
                end = time.time()
                elapsed_time_lgpmin = end - start
                n_data = lgpmin_opt.model_calculator.max_data
                size_cov_lgpmin = ((len(atoms) * 3) * n_data + n_data)**2
                print('\n Time LGPMin: ', elapsed_time_lgpmin)
                print('Data mode:', data_mode)
                print('Max_data:', max_data)
                print('Size covariance LGPMin: ', size_cov_lgpmin)
                print('Feval LGPMin:', lgpmin_opt.function_calls)
                results['cluster_size'].append(cluster_size)
                results['seed'].append(seed)
                results['algorithm'].append('LGPMin')
                results['max_data'].append(n_data)
                results['mode_max_data'].append(data_mode)
                results['energy'].append(initial_lgpmin.get_potential_energy())
                results['feval'].append(lgpmin_opt.function_calls)
                results['elapsed_time'].append(elapsed_time_lgpmin)
                results['covariance_size'].append(size_cov_lgpmin)

                df = pd.DataFrame(results)
                df.to_csv('results.csv')

