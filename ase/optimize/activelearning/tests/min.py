from math import cos, sin, pi

import numpy as np

import ase.units as units
from ase import Atoms
from ase.calculators.tip3p import TIP3P, rOH, angleHOH
from ase.constraints import FixInternals

from ase.optimize.test.test import Wrapper

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
        print('%d force calls' % wrapper.nsteps)
        fmax = (self.wrapper.get_forces() ** 2).sum(axis=1).max()
        e = self.wrapper.get_potential_energy()
        self.f.append(np.sqrt(fmax))
        self.e.append(e)
        

r = rOH
a = angleHOH * pi / 180

# From http://dx.doi.org/10.1063/1.445869
eexp = 6.50 * units.kcal / units.mol
dexp = 2.74
aexp = 27

D = np.linspace(2.5, 3.5, 30)

calc = TIP3P()
optimizers = [OldGPMin, GPMin]

results = {}
for i,optimizer in enumerate(optimizers):
    dimer = Atoms('H2OH2O',
                  [(r * cos(a), 0, r * sin(a)),
                   (r, 0, 0),
                   (0, 0, 0),
                   (r * cos(a / 2), r * sin(a / 2), 0),
                   (r * cos(a / 2), -r * sin(a / 2), 0),
                   (0, 0, 0)])
    dimer.calc = calc

    E = []
    F = []
    for d in D:
        dimer.positions[3:, 0] += d - dimer.positions[5, 0]
        E.append(dimer.get_potential_energy())
        F.append(dimer.get_forces())

    F = np.array(F)

    F1 = np.polyval(np.polyder(np.polyfit(D, E, 7)), D)
    F2 = F[:, :3, 0].sum(1)
    error = abs(F1 - F2).max()

    dimer.constraints = FixInternals(
        bonds=[(r, (0, 2)), (r, (1, 2)),
               (r, (3, 5)), (r, (4, 5))],
        angles=[(a, (0, 2, 1)), (a, (3, 5, 4))])

    wrapper = Wrapper(dimer)
    tracker = Tracker(wrapper)

    opt = optimizer(wrapper, update_hyperparams=True)
    opt.attach(tracker)
    opt.run(0.01)

    #Check the result is the same
    results[i] = {'e':np.array(tracker.e), 'f':np.array(tracker.f)}


assert np.allclose(results[0]['e'], results[1]['e'])
assert np.allclose(results[0]['f'], results[1]['f'], rtol=0.01)
