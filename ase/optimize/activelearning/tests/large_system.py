import numpy as np
from ase.cluster import wulff_construction
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms

from ase.optimize.activelearning.oldgpmin import OldGPMin
from ase.optimize.activelearning.aidmin import GPMin

class Tracker:
    """
    Observer to keep track of details of the
    optimization on the fly.
    """
    def __init__(self, wrapper):
        self.wrapper = wrapper
        self.f = []
        self.e = []

    def __call__(self):
        fmax = (self.wrapper.get_forces() ** 2).sum(axis=1).max()
        e = self.wrapper.get_potential_energy()
        self.f.append(np.sqrt(fmax))
        self.e.append(e)


# Define system
surfaces = [(1, 0, 0), (1, 1, 0), (1, 1, 1)]
esurf = [0.9151, 0.9771, 0.7953] # Surface energies
size = 79    # number of atoms
atoms = wulff_construction('Au', surfaces, esurf, size, 'fcc',
                                 rounding = 'above')
atoms.center(vacuum=6)

# Set some not very reallistic constraints
mask = [a%5 == 0 for a in range(size)]
constraint = FixAtoms(mask=mask)
atoms.set_constraint(constraint)

atoms.rattle(0.2)

optimizers = [OldGPMin, GPMin]

results = {}
for i, optimizer in enumerate(optimizers):
    atoms0 = atoms.copy()
    atoms0.set_calculator(EMT())

    tracker = Tracker(atoms0)

    opt = optimizer(atoms0)
    opt.attach(tracker)
    opt.run(0.01)


    #Check the result is the same
    results[i] = {'e':np.array(tracker.e), 'f':np.array(tracker.f)}
