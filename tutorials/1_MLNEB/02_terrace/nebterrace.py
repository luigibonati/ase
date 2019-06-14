from ase.calculators.emt import EMT
from ase.build import fcc211, add_adsorbate
from ase.constraints import FixAtoms
from ase.optimize import BFGS, MDMin
from ase.neb import NEB, NEBTools
from ase.io import read
from ase.optimize.bayesian.mlneb import MLNEB

import matplotlib.pyplot as plt
import numpy as np

""" 
    Toy model for the diffusion of a Au atom on an Pt(211) surface.   
"""

# Define number of images:
n_images = 15

# 1. Structural relaxation.

# Setup calculator:
ase_calculator = EMT()

# Pt atom adsorbed in a hollow site:
slab = fcc211('Pt', size=(6, 6, 3), vacuum=4.0)
add_adsorbate(slab, 'Au', 0.5, (-0.1, 2.7))

# Fix bottom layers.
atom_indices = np.arange(6, 12).tolist()
atom_indices_2 = np.arange(18, 24).tolist()
atom_indices_3 = np.arange(30, len(slab)-1).tolist()
index_fixed_atoms = atom_indices + atom_indices_2 + atom_indices_3

slab.set_constraint(FixAtoms(index_fixed_atoms))

# Use EMT potential:
slab.set_calculator(EMT())

# Optimize initial state:
qn = BFGS(slab, trajectory='initial_opt.traj')
qn.run(fmax=0.02)

# Optimize final state:
slab[-1].x = 9.053
slab[-1].y += 6.930
qn = BFGS(slab, trajectory='final_opt.traj')
qn.run(fmax=0.02)


# 1. A. NEB.
initial_ase = read('initial_opt.traj')
final_ase = read('final_opt.traj')

images_ase = [initial_ase]
for i in range(1, n_images-1):
    image = initial_ase.copy()
    image.set_calculator(EMT())
    images_ase.append(image)
images_ase.append(final_ase)

neb_ase = NEB(images_ase, climb=True, method='improvedtangent')
neb_ase.interpolate(method='idpp', mic=False)
qn_ase = MDMin(neb_ase, trajectory='neb_ase.traj')
qn_ase.run(fmax=0.05)

# 2.B. ML-NEB.
mlneb = MLNEB(start='initial_opt.traj',
                 end='final_opt.traj',
                 ase_calc=EMT(),
                 n_images=n_images,
                 interpolation='idpp',
                 restart=None)
mlneb.run(fmax=0.05, trajectory='ML-NEB.traj')

# Plot ASE NEB.
nebtools = NEBTools(images_ase)
nebtools.plot_band()
plt.show()

# Plot ML-NEB predicted path.
nebtools_mlneb = NEBTools(mlneb.images)
S_mlneb, E_mlneb, Sf_ase, Ef_ase, lines = nebtools_mlneb.get_fit()
Ef_neb_ase, dE_neb_ase = nebtools_mlneb.get_barrier(fit=False)
Ef_mlneb, dE_mlneb = nebtools_mlneb.get_barrier(fit=False)
fig, ax = plt.subplots()

uncertainty_neb = []
for i in mlneb.images:
    uncertainty_neb += [i.info['uncertainty']/2.0]

# Add predictions' uncertainty to the plot.
ax.errorbar(S_mlneb, E_mlneb, uncertainty_neb, alpha=0.8,
            markersize=0.0, ecolor='midnightblue', ls='',
            elinewidth=3.0, capsize=1.0)
nebtools_mlneb.plot_band(ax=ax)
plt.show()
