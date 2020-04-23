import numpy as np

from math import exp, sqrt

from ase.calculators.calculator import Calculator
from ase.neighborlist import neighbor_list

class MorsePotential(Calculator):
    """Morse potential.

    Default values chosen to be similar as Lennard-Jones.
    """

    implemented_properties = ['energy', 'forces']
    default_parameters = {'epsilon': 1.0,
                          'rho0': 6.0,
                          'r0': 1.0,
                          'rc': None}
    nolabel = True

    def __init__(self, **kwargs):
        Calculator.__init__(self, **kwargs)

    def calculate(self, atoms=None, properties=['energy'],
                  system_changes=['positions', 'numbers', 'cell',
                                  'pbc', 'charges', 'magmoms']):
        Calculator.calculate(self, atoms, properties, system_changes)
        epsilon = self.parameters.epsilon
        rho0 = self.parameters.rho0
        r0 = self.parameters.r0
        rc = self.parameters.rc
        if rc is None:
            rc = 3 * r0

        energy = 0.0
        forces = np.zeros((len(self.atoms), 3))
        preF = 2 * epsilon * rho0 / r0

        I, J, d, D = neighbor_list('ijdD', atoms, rc)
        for (i1, i2, r, diff) in zip(I, J, d, D):
            expf = exp(rho0 * (1.0 - r / r0))
            energy += epsilon * expf * (expf - 2)
            F = preF * expf * (expf - 1) * diff / r
            forces[i1] -= F
            forces[i2] += F
        self.results['energy'] = energy
        self.results['forces'] = forces
