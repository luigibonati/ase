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
        """
        Parameters
        ----------
        epsilon: float
          Absolute minimum depth, default 1.0
        r0: float
          Minimum distance, default 1.0
        rho0: float
          Exponential prefactor. The force constant in the potential minimum
          is k = 2 * epsilon * (rho0 / r0)**2, default 6.0
        """
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

        forces = np.zeros((len(self.atoms), 3))
        preF = - 2 * epsilon * rho0 / r0

        i, j, d, D = neighbor_list('ijdD', atoms, rc)
        expf = np.exp(rho0 * (1.0 - d/r0))
        energy = 0.5*(epsilon * expf * (expf - 2)).sum()

        F = preF * (expf * (expf - 1) * (D / d[:, None]).T).T
        for dim in range(3):
            forces[:, dim] = np.bincount(i, weights=F[:, dim],
                                         minlength=len(atoms))

        self.results['energy'] = energy
        self.results['forces'] = forces
