import numpy as np
from scipy.linalg import cho_solve
from scipy.stats import invgamma
from scipy.special import expit
import warnings
from ase.calculators.calculator import PropertyNotImplementedError

from ase.calculators.calculator import Calculator, all_changes
from ase.neighborlist import NeighborList
from ase.data import covalent_radii


class Prior():
    '''Base class for all priors for the bayesian optimizer.

       The __init__ method and the prior method are implemented here.
       Each child class should implement its own potential method,
       that will be called by the prior method implemented here.

       When used, the prior should be initialized outside the optimizer
       and the Prior object should be passed as a function to the optimizer.
    '''

    def __init__(self, **kwargs):
        '''Basic prior implementation.
        '''

        # By default, do not let the prior use the update method
        self.use_update = False

        self.use_forces = kwargs.get('use_forces')
        if self.use_forces is None:
            self.use_forces = True

    def prior(self, x):
        ''' Actual prior function, common to all Priors.

        Parameters:

        x: Fingerprint object '''
        return self.potential(x)

    def let_update(self):
        if hasattr(self, 'update'):
            self.use_update = True
        else:
            warning = ('The prior does not have implemented an update method ',
                       'the prior has thus not been updated')
            warnings.warn(warning)


class ZeroPrior(Prior):
    '''ZeroPrior object, consisting on a constant prior with 0eV energy.'''

    def __init__(self, **kwargs):
        Prior.__init__(self, **kwargs)

    def potential(self, x):
        if self.use_forces:
            d = len(x.atoms) * 3  # number of forces
            output = np.zeros(d + 1)
        else:
            output = np.zeros(1)
        return output


class ConstantPrior(Prior):
    '''Constant prior, with energy = constant and zero forces

    Parameters:

    constant: energy value for the constant.
    '''

    def __init__(self, constant, **kwargs):
        self.constant = constant
        Prior.__init__(self, **kwargs)

    def potential(self, x):
        if self.use_forces:
            d = len(x.atoms) * 3  # number of forces
            output = np.zeros(d + 1)
        else:
            output = np.zeros(1)
        output[0] = self.constant
        return output

    def set_constant(self, constant):
        self.constant = constant

    def update(self, X, Y, L):
        """Update the constant to maximize the marginal likelihood.

        The optimization problem:
        m = argmax [-1/2 (y-m).T K^-1(y-m)]

        can be turned into an algebraic problem
        m = [ u.T K^-1 y]/[u.T K^-1 u]

        where u is the constant prior with energy 1 (eV).

        parameters:
        ------------
        y: training targets
        L: Cholesky factor of the kernel """

        # Get derivative of prior respect to constant: we call it u
        self.set_constant(1.)
        u = np.hstack([self.potential(x) for x in X])

        # w = K\u
        w = cho_solve((L, True), u, check_finite=False)

        # Set constant
        m = np.dot(w, np.array(Y).flatten()) / np.dot(w, u)
        self.set_constant(m)


class CalculatorPrior(ConstantPrior):

    '''CalculatorPrior object, allows the user to
    use another calculator as prior function instead of the
    default constant.

    The form of prior is

    E_p(x) = m*u + E_c(x)

    where m is constant (energy), u is array with 1's for energy
    and 0 for force components, E_c(x) is the calculator
    potential.

    Parameters:

    calculator: one of ASE's calculators
    **kwargs: arguments passed to ASE parent calculator
    '''

    def __init__(self, calculator, **kwargs):

        Prior.__init__(self, **kwargs)
        self.calculator = calculator
        self.constant = 0.0  # baseline

    def potential(self, x):

        if self.use_forces:
            d = len(x.atoms) * 3  # number of forces
            output = np.zeros(d + 1)
        else:
            output = np.zeros(1)

        atoms = x.atoms.copy()

        atoms.calc = self.calculator
        output[0] = atoms.get_potential_energy() + self.constant

        if self.use_forces:
            try:
                output[1:] = -atoms.get_forces().reshape(-1)
            except PropertyNotImplementedError:
                # warning = 'Prior Calculator does not support forces. '
                # warning += 'Setting all prior forces to zero.'
                # warnings.warn(warning)
                print('Prior Calculator does not support forces.')

        return output

    def update(self, X, Y, L):
        """Update the constant to maximize the marginal likelihood.

        The optimization problem:
        m = argmax [-1/2 (y-m-Ec).T K^-1(y-m-Ec)]

        can be turned into an algebraic problem
        m = [ u.T K^-1 (y - Ec)]/[u.T K^-1 u]

        where u is the constant prior with energy 1 (eV),
        and Ec is the calculator prior.

        parameters:
        ------------
        X: training parameters
        Y: training targets
        L: Cholesky factor of the kernel """

        # Get derivative of prior respect to constant: we call it u
        self.set_constant(1.)
        u = np.hstack([ConstantPrior.potential(self, x) for x in X])

        E_calcprior = (np.hstack([self.potential(x) for x in X]) - u).flatten()

        # w = K\u
        w = cho_solve((L, True), u, check_finite=False)

        # Set constant
        m = np.dot(w, (np.array(Y).flatten() - E_calcprior)) / np.dot(w, u)
        self.set_constant(m)


class RepulsivePotential(Calculator):
    ''' Repulsive potential of the form
    sum_ij (0.7 * (Ri + Rj) / rij)**12

    where Ri and Rj are the covalent radii of atoms
    '''

    implemented_properties = ['energy', 'forces']
    default_parameters = {'prefactor': 0.7,
                          'rc': 1.0}
    nolabel = True

    def __init__(self, **kwargs):
        Calculator.__init__(self, **kwargs)

    def calculate(self, atoms=None,
                  properties=['energy', 'forces'],
                  system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)

        self.atoms = atoms.copy()
        natoms = len(self.atoms)

        prefactor = self.parameters.prefactor
        rc = self.parameters.rc

        if 'numbers' in system_changes:
            self.nl = NeighborList([rc / 2] * natoms, self_interaction=False)
        self.nl.update(self.atoms)

        positions = self.atoms.positions
        cell = self.atoms.cell

        energy = 0.0
        forces = np.zeros((natoms, 3))

        for a1 in range(natoms):
            neighbors, offsets = self.nl.get_neighbors(a1)
            cells = np.dot(offsets, cell)
            d = positions[neighbors] + cells - positions[a1]
            r2 = (d**2).sum(1)

            # covalent radius of a1:
            cr1 = covalent_radii[self.atoms[a1].number]

            # covalent radii of neighbors:
            crs = np.array([covalent_radii[self.atoms[i].number]
                            for i in neighbors])

            c2 = (crs + cr1)**2 / r2
            c2[r2 > rc**2] = 0.0
            c12 = prefactor**12 * c2**6
            energy += c12.sum()

            # Forces:
            f = (12 * c12 / r2)[:, np.newaxis] * d
            forces[a1] -= f.sum(axis=0)
            for a2, f2 in zip(neighbors, f):
                forces[a2] += f2

        self.results['energy'] = energy
        self.results['forces'] = forces


class PriorDistribution():
    
    def __init__(self, mean, std):
        ''' Gaussian Prior distribution for hyperparameter 'scale' '''
        self.mean = mean
        self.std = std

    def get(self, gp):
        scale = gp.hyperparams['scale']
        log_prefactor = np.log(1 / np.sqrt(2 * np.pi) / self.std)
        log_exp = -(scale - self.mean)**2 / 2 / self.std**2
        print(log_prefactor + log_exp)
        return (log_prefactor + log_exp)


class PriorDistributionInvGamma():
    
    def __init__(self, a, loc, scale):
        ''' Inverse Gamma function Prior distribution for
        hyperparameter 'scale'. Parameters correspond to
        parameters of scipy.stats.invgamma. '''

        self.a = a
        self.loc = loc
        self.scale = scale  # inverse gamma parameters

    def get(self, gp):
        
        scale = gp.hyperparams['scale']
        value = np.log(invgamma.pdf(scale, self.a,
                                    self.loc, self.scale))
        return value


class PriorDistributionSigmoid():
    
    def __init__(self, loc, width):
        ''' Sigmoid Prior "distribution" for
        hyperparameter 'scale'. Parameters correspond to
        parameters of scipy.special.expit. '''

        self.loc = loc
        self.width = width

    def get(self, gp):
        
        scale = gp.hyperparams['scale']
        x = (scale - self.loc) / self.width
        value = np.log(expit(x))
        return value
