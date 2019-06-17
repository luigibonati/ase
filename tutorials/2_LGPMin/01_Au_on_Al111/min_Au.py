from ase.build import fcc100, add_adsorbate
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms
from ase.optimize.bayesian.lgpmin import LGPMin
from ase.optimize.bayesian.model import GPModel
from ase.optimize.gpmin.gpmin import GPMin
import time
""" 
    Toy model for the optimization of a Au atom on an Al(111) surface.  
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
slab.set_calculator(EMT())
# 1.2. Optimize Atoms structure:

# Set your own model parameters.
gp_model = GPModel(scale=0.4, weight=1.0, noise=0.005,
                   update_hyperparams=False)

lgpmin = LGPMin(atoms=slab, model=gp_model)
start = time.time()
lgpmin.run(fmax=0.02, restart=False)
end = time.time()
print('Time optimization LGPMin:', end-start)


# 2x2-Al(001) surface with 3 layers and an
# Au atom adsorbed in a hollow site:
slab = fcc100('Al', size=(2, 2, 3))
add_adsorbate(slab, 'Au', 1.7, 'hollow')
slab.center(axis=2, vacuum=4.0)
slab.set_calculator(EMT())

# Fix second and third layers:
mask = [atom.tag > 1 for atom in slab]
slab.set_constraint(FixAtoms(mask=mask))
slab.set_calculator(EMT())
# 1.2. Optimize Atoms structure:

opt = GPMin(slab)
start = time.time()
opt.run(fmax=0.02)
end = time.time()
print('Time optimization GPMin:', end-start)
