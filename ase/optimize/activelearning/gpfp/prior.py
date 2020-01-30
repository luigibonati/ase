import numpy as np
from scipy.linalg import cho_solve

import warnings


class Prior():
    '''Base class for all priors for the bayesian optimizer.

       The __init__ method and the prior method are implemented here.
       Each child class should implement its own potential method,
       that will be called by the prior method implemented here.

       When used, the prior should be initialized outside the optimizer
       and the Prior object should be passed as a function to the optimizer.
    '''

    def __init__(self):
        '''Basic prior implementation.
        '''

        # By default, do not let the prior use the update method
        self.use_update = False

    def prior(self, x):
        ''' Actual prior function, common to all Priors'''

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

    def __init__(self):
        Prior.__init__(self)

    def potential(self, x):
        return np.zeros(x.shape[0] + 1)


class ConstantPrior(Prior):
    '''Constant prior, with energy = constant and zero forces

    Parameters:

    constant: energy value for the constant.
    '''

    def __init__(self, constant, use_forces=True):
        self.constant = constant
        self.use_forces = use_forces
        Prior.__init__(self)

    def potential(self, x):
        d = x.shape[0]
        if self.use_forces:
            output = np.zeros(d + 1)
        else:
            output = np.zeros(1)
        output[0] = self.constant
        return output

    def set_constant(self, constant):
        self.constant = constant

    def update(self, x, y, L):
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
        u = self.prior(x)

        # w = K\u
        w = cho_solve((L, True), u, check_finite=False)

        # Set constant
        m = np.dot(w, y.flatten()) / np.dot(w, u)
        self.set_constant(m)


class CalculatorPrior(Prior):

    '''CalculatorPrior object, allows the user to
    use another calculator as prior function instead of the
    default constant.

    Parameters:

    atoms: the Atoms object
    calculator: one of ASE's calculators

    '''

    def __init__(self, calculator, use_forces=True):

        Prior.__init__(self)
        self.calculator = calculator
        self.use_forces = True

    def potential(self, x):

        d = len(x.atoms) * 3
        if self.use_forces:
            output = np.zeros(d + 1)
        else:
            output = np.zeros(1)

        x.atoms.set_calculator(self.calculator)
        output[0] = x.atoms.get_potential_energy()

        return output
