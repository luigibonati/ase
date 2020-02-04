import numpy as np
from scipy.linalg import cho_solve
import warnings
from ase.calculators.calculator import PropertyNotImplementedError


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


class CalculatorPrior(Prior):

    '''CalculatorPrior object, allows the user to
    use another calculator as prior function instead of the
    default constant.

    Parameters:

    atoms: the Atoms object
    calculator: one of ASE's calculators

    TODO: Add update method
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

        x.atoms.set_calculator(self.calculator)
        output[0] = x.atoms.get_potential_energy() + self.constant

        if self.use_forces:
            try:
                output[1:] = -x.atoms.get_forces().reshape(-1)
            except PropertyNotImplementedError:
                # warning = 'Prior Calculator does not support forces. '
                # warning += 'Setting all prior forces to zero.'
                # warnings.warn(warning)
                pass

        return output
