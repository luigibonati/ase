from ase.build import fcc100, add_adsorbate
from ase.calculators.emt import EMT
from ase.io import read
from ase.constraints import FixAtoms
from ase.neb import NEB
from ase.optimize import BFGS
import matplotlib.pyplot as plt
from ase.optimize.mlneb import MLNEB
from ase.neb import NEBTools

""" 
    Toy model for the diffusion of a Au atom on an Al(111) surface.  
    This example contains: 
    1. Optimization of the initial and final end-points of the reaction path. 
    2.A. NEB optimization using CI-NEB. 
    2.B. NEB optimization using a machine-learning surrogate model.
    3. Comparison between the CI-NEB and the ML-NEB algorithm.
"""

# 1. Structural relaxation.

# 1.1. Structures:

# 2x2-Al(001) surface with 3 layers and an
# Au atom adsorbed in a hollow site:
slab = fcc100('Al', size=(2, 2, 3))
add_adsorbate(slab, 'Au', 1.7, 'hollow')
slab.center(axis=2, vacuum=4.0)
slab.set_calculator(EMT())

# Fix second and third layers:
mask = [atom.tag > 1 for atom in slab]
slab.set_constraint(FixAtoms(mask=mask))

# 1.2. Optimize initial and final end-points.

# Initial end-point:
qn = BFGS(slab, trajectory='initial.traj')
qn.run(fmax=0.01)

# Final end-point:
slab[-1].x += slab.get_cell()[0, 0] / 2
qn = BFGS(slab, trajectory='final.traj')
qn.run(fmax=0.01)

# Set number of images:
n_images = 11

# 2.A. Original CI-NEB.
initial_ase = read('initial.traj')
final_ase = read('final.traj')
constraint = FixAtoms(mask=[atom.tag > 1 for atom in initial_ase])

images_ase = [initial_ase]
for i in range(1, n_images-1):
    image = initial_ase.copy()
    image.set_calculator(EMT())
    image.set_constraint(constraint)
    images_ase.append(image)

images_ase.append(final_ase)

neb_ase = NEB(images_ase, climb=True)
neb_ase.interpolate(method='idpp')

qn_ase = BFGS(neb_ase, trajectory='neb_ase.traj')
qn_ase.run(fmax=0.05)

# 2.B. ML-NEB.
mlneb = MLNEB(start='initial.traj', end='final.traj', restart=None,
              ase_calc=EMT(), interpolation='idpp', n_images=n_images)
mlneb.run(fmax=0.05, trajectory='ML-NEB.traj')

# 3. Summary of the results.

# CI-NEB function evaluations:
print('\nSummary of the results: \n')
atoms_ase = read('neb_ase.traj', ':')
n_eval_ase = int(len(atoms_ase) - 2 * (len(atoms_ase)/n_images))
print('Number of function evaluations CI-NEB:', n_eval_ase)

# ML-NEB Function evaluations:
n_eval_mlneb = mlneb.function_calls
print('Number of function evaluations ML-NEB:', n_eval_mlneb)

# Plot ASE NEB.
nebtools_mlneb = NEBTools(images_ase)
nebtools_mlneb.plot_band()
plt.show()

# Plot ML-NEB predicted path.
nebtools_mlneb = NEBTools(mlneb.images)
S_mlneb, E_mlneb, Sf_ase, Ef_ase, lines = nebtools_mlneb.get_fit()
Ef_neb_ase, dE_neb_ase = nebtools_mlneb.get_barrier(fit=False)
Ef_mlneb, dE_mlneb = nebtools_mlneb.get_barrier(fit=False)
fig, ax = plt.subplots()

uncertainty_neb = []
for i in mlneb.images:
    uncertainty_neb += [i.info['uncertainty']]

# Add predictions' uncertainty to the plot.
ax.errorbar(S_mlneb, E_mlneb, uncertainty_neb, alpha=0.8,
            markersize=0.0, ecolor='red', ls='',
            elinewidth=3.0, capsize=1.0)
nebtools_mlneb.plot_band(ax=ax)
plt.show()
