from ase.calculators.emt import EMT
from ase.io import read
from ase.neb import NEB
from ase.optimize import BFGS, MDMin
import matplotlib.pyplot as plt
from ase.optimize.bayesian.gpneb import GPNEB
from ase.neb import NEBTools
from ase.build import fcc111, add_adsorbate
from ase.constraints import FixAtoms


""" 
    Toy model rearrangement of Pt heptamer island on Pt(111).
"""

# 1. Structural relaxation.

# Setup calculator:
ase_calculator = EMT()

# 1.1. Structures:

# Build a 3 layers 5x5-Pt(111) slab.
atoms = fcc111('Pt', size=(5, 5, 3))
c = FixAtoms(indices=[atom.index for atom in atoms if atom.symbol == 'Pt' and atom.position[2]<2.3])
atoms.set_constraint(c)

atoms.center(axis=2, vacuum=15.0)

# Build heptamer island:
atoms2 = fcc111('Au', size=(3, 3, 1))
atoms2.pop(0)
atoms2.pop(7)
atoms2.rattle(stdev=0.10, seed=0)

# Add island to slab:
add_adsorbate(atoms, atoms2, 2.5, offset=0.5)
atoms.set_calculator(EMT())

# 1.2. Optimize initial and final end-points.
# Initial end-point:
qn = BFGS(atoms, trajectory='initial.traj')
qn.run(fmax=0.01)

# Final end-point:
atoms.positions[80][0] += 2.772
atoms.positions[81][0] += 2.772
qn = BFGS(atoms, trajectory='final.traj')
qn.run(fmax=0.01)

# Set number of images
n_images = 11

# 2.A. Original CI-NEB.
initial_ase = read('initial.traj')
final_ase = read('final.traj')

ase_calculator = EMT()

images_ase = [initial_ase]
for i in range(1, n_images-1):
    image = initial_ase.copy()
    image.set_calculator(EMT())
    images_ase.append(image)

images_ase.append(final_ase)

neb_ase = NEB(images_ase, climb=True)
neb_ase.interpolate(method='idpp')

# qn_ase = MDMin(neb_ase, trajectory='neb_ase.traj')
# qn_ase.run(fmax=0.05)

# 2.B. GPNEB.
gpneb = GPNEB(start='initial.traj', end='final.traj',
              calculator=EMT(),
              n_images=n_images,
              interpolation='idpp')

gpneb.run(fmax=0.05, trajectory='GPNEB.traj')

# 3. Summary of the results.

# NEB ASE function evaluations:
print('\nSummary of the results: \n')
atoms_ase = read('neb_ase.traj', ':')
n_eval_ase = int(len(atoms_ase) - 2 * (len(atoms_ase)/n_images))
print('Number of function evaluations CI-NEB:', n_eval_ase)

# GPNEB Function evaluations:
n_eval_gpneb = gpneb.function_calls
print('Number of function evaluations GPNEB:', n_eval_gpneb)

# Plot ASE NEB.
nebtools_gpneb = NEBTools(images_ase)
nebtools_gpneb.plot_band()
plt.show()

# Plot GPNEB predicted path.
nebtools_gpneb = NEBTools(gpneb.images)
S_gpneb, E_gpneb, Sf_ase, Ef_ase, lines = nebtools_gpneb.get_fit()
Ef_neb_ase, dE_neb_ase = nebtools_gpneb.get_barrier(fit=False)
Ef_gpneb, dE_gpneb = nebtools_gpneb.get_barrier(fit=False)
fig, ax = plt.subplots()

uncertainty_neb = []
for i in gpneb.images:
    uncertainty_neb += [i.info['uncertainty']/2.0]

# Add predictions' uncertainty to the plot.
ax.errorbar(S_gpneb, E_gpneb, uncertainty_neb, alpha=0.8,
            markersize=0.0, ecolor='midnightblue', ls='',
            elinewidth=3.0, capsize=1.0)
nebtools_gpneb.plot_band(ax=ax)
plt.show()
