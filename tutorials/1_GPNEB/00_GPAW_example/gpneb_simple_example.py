from ase.build import fcc100, add_adsorbate
from ase.io import read
from ase.constraints import FixAtoms
from ase.optimize import BFGS
from ase.optimize.activelearning.gpneb import GPNEB
from ase.optimize.activelearning.model import GPModel
from gpaw import GPAW
from ase.visualize import view

""" 
    Simple tutorial to set up a GPNEB calculation using GPAW.
"""
# 0. Set up GPAW settings:
calc_args = {'mode': 'lcao', 'basis': 'sz(dzp)', 'h': 0.3,
             'txt': 'gpaw.txt', 'parallel': dict(sl_auto=False),
             'convergence': dict(density=1e-4)}

# 1. Structural relaxation.

# 1.1. Optimize initial NEB end-point.
slab = fcc100('Al', size=(2, 2, 3))
add_adsorbate(slab, 'Au', 1.7, 'hollow')
slab.center(axis=2, vacuum=4.0)
slab.set_calculator(GPAW(**calc_args))
mask = [atom.tag > 1 for atom in slab]
slab.set_constraint(FixAtoms(mask=mask))
qn = BFGS(slab, trajectory='initial.traj')
qn.run(fmax=0.05)

# 1.2. Optimize final NEB end-point.
slab = fcc100('Al', size=(2, 2, 3))
add_adsorbate(slab, 'Au', 1.7, 'hollow')
slab.center(axis=2, vacuum=4.0)
slab.set_calculator(GPAW(**calc_args))
mask = [atom.tag > 1 for atom in slab]
slab.set_constraint(FixAtoms(mask=mask))
slab[-1].x = 4.3
qn = BFGS(slab, trajectory='final.traj')
qn.run(fmax=0.05)

# 2. Obtain minimum energy pathway using GPNEB.

# 2.1. Optional: Select settings for the predictive model using the GPModel class.
gp_model = GPModel(scale=0.35, weight=1.0, noise=0.001, update_hyperparams=False)

# 2.2. Feed the previously optimized initial and final end-points and your model to the GPNEB class.
gpneb = GPNEB(start='initial.traj', end='final.traj',  # Initial and final end-points.
              model=gp_model,  # Model with personalized GP parameters. If None, default parameters.
              calculator=GPAW(**calc_args),  # ASE Calculator including settings.
              interpolation='idpp',  # Interpolation method.
              n_images=7  # Number of images
              )

# 2.3. Run the algorithm.
gpneb.run(fmax=0.05, trajectory='GPNEB.traj')

# 2.4. Visualize the optimized path.
optimized_images = read('GPNEB.traj', '-7:')  # Read the 7 last predicted images.
view(optimized_images)

# 2.5. The uncertainty of each image can be accessed using info.
for i in range(0, len(optimized_images)):
    print('Uncertainty on image ' + str(i) + ': ' + optimized_images[i].info['uncertainty'])

# 2.6. The evaluated structures (experiences) to build the model are found in GPNEB_experiences.traj
experiences = read('GPNEB_experiences.traj', ':')
view(experiences)