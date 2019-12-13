from math import cos, sin, pi

import numpy as np

import ase.units as units
from ase import Atoms
from ase.calculators.tip3p import TIP3P, epsilon0, sigma0, rOH, angleHOH
from ase.calculators.qmmm import (SimpleQMMM, EIQMMM, LJInteractions,
                                  LJInteractionsGeneral)
from ase.constraints import FixInternals

import ase.optimize.activelearning as aid
from ase.optimize.test.test import Wrapper

# Observer to print the number of steps on the fly
class NstepPrinter:
    def __init__(self, wrapper):
        self.wrapper = wrapper
    def __call__(self):
        print('%d force calls' % wrapper.nsteps)
        print('calculator: '+wrapper.calc.__class__.__name__)

r = rOH
a = angleHOH * pi / 180

# From http://dx.doi.org/10.1063/1.445869
eexp = 6.50 * units.kcal / units.mol
dexp = 2.74
aexp = 27

D = np.linspace(2.5, 3.5, 30)

i = LJInteractions({('O', 'O'): (epsilon0, sigma0)})

# General LJ interaction object
sigma_mm = np.array([0, 0, sigma0])
epsilon_mm = np.array([0, 0, epsilon0])
sigma_qm = np.array([0, 0, sigma0])
epsilon_qm = np.array([0, 0, epsilon0])
ig = LJInteractionsGeneral(sigma_qm, epsilon_qm, sigma_mm, epsilon_mm, 3)

calc = TIP3P()
optimizers = [aid.oldgpmin.OldGPMin, aid.aidmin.GPMin]

for optimizer in optimizers:
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
    countsteps = NstepPrinter(wrapper)

    opt = optimizer(wrapper)
    opt.attach(countsteps)
    opt.run(0.01)
