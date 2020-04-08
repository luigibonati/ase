import re
import numpy as np
import numpy.linalg as la
from itertools import product, combinations


class Kernel():
    def __init__(self):
        pass

    def set_params(self, params):
        pass

    def kernel(self, x1, x2):
        """Kernel function to be fed to the Kernel matrix"""
        pass

    def K(self, X1, X2):
        """Compute the kernel matrix """
        return np.block([[self.kernel(x1, x2) for x2 in X2] for x1 in X1])


class SE_kernel(Kernel):
    """Squared exponential kernel without derivatives"""
    def __init__(self):
        Kernel.__init__(self)

    def set_params(self, params):
        """Set the parameters of the squared exponential kernel.

        Parameters:

        params: (dictionary) Parameters of the kernel:
            {'weight': prefactor of the exponential,
             'scale' : scale of the kernel}
        """

        if not hasattr(self, 'params'):
            self.params = {}
        for p in params:
            self.params[p] = params[p]

        self.weight = self.params.get('weight', None)
        self.l = self.params.get('scale', None)

        if self.weight is None or self.l is None:
            raise ValueError('The parameters of the kernel have not been set')

    def squared_distance(self, x1, x2):
        """Returns the norm of x1-x2 using diag(l) as metric """
        return np.sum((x1 - x2) * (x1 - x2)) / self.l**2

    def kernel(self, x1, x2):
        """ This is the squared exponential function"""
        return self.weight**2 * np.exp(-0.5 * self.squared_distance(x1, x2))

    def dK_dweight(self, x1, x2):
        """Derivative of the kernel respect to the weight """
        return 2 * self.weight * np.exp(-0.5 * self.squared_distance(x1, x2))

    def dK_dl(self, x1, x2):
        """Derivative of the kernel respect to the scale"""
        return self.kernel * la.norm(x1 - x2)**2 / self.l**3


class SquaredExponential(SE_kernel):
    """Squared exponential kernel with derivatives.
    For the formulas see Koistinen, Dagbjartsdottir, Asgeirsson, Vehtari,
    Jonsson.
    Nudged elastic band calculations accelerated with Gaussian process
    regression. Section 3.

    Before making any predictions, the parameters need to be set using the
    method SquaredExponential.set_params(params) where the parameters are a
    list whose first entry is the weight (prefactor of the exponential) and
    the second is the scale (l).

    Parameters:

    dimensionality: The dimensionality of the problem to optimize, typically
                    3*N where N is the number of atoms. If dimensionality is
                    None, it is computed when the kernel method is called.

    Attributes:
    ----------------
    D:          int. Dimensionality of the problem to optimize
    weight:     float. Multiplicative constant to the exponenetial kernel
    l :         float. Length scale of the squared exponential kernel

    Relevant Methods:
    ----------------
    set_params:         Set the parameters of the Kernel, i.e. change the
                        attributes
    kernel_function:    Squared exponential covariance function
    kernel:             covariance matrix between two points in the manifold.
                        Note that the inputs are arrays of shape (D,)
    kernel_matrix:      Kernel matrix of a data set to itself, K(X,X)
                        Note the input is an array of shape (nsamples, D)
    kernel_vector       Kernel matrix of a point x to a dataset X, K(x,X).

    gradient:           Gradient of K(X,X) with respect to the parameters of
                        the kernel i.e. the hyperparameters of the Gaussian
                        process.
    """

    def __init__(self, dimensionality=None):
        self.D = dimensionality
        SE_kernel.__init__(self)

    def kernel_function(self, x1, x2):
        """ This is the squared exponential function"""
        return self.weight**2 * np.exp(-0.5 * self.squared_distance(x1, x2))

    def kernel_function_gradient(self, x1, x2):
        """Gradient of kernel_function respect to the second entry.
        x1: first data point
        x2: second data point
        """
        prefactor = (x1 - x2) / self.l**2
        # return prefactor * self.kernel_function(x1,x2)
        return prefactor

    def kernel_function_hessian(self, x1, x2):
        """Second derivatives matrix of the kernel function"""
        P = np.outer(x1 - x2, x1 - x2) / self.l**2
        prefactor = (np.identity(self.D) - P) / self.l**2
        return prefactor

    def kernel(self, x1, x2):
        """Squared exponential kernel including derivatives.
        This function returns a D+1 x D+1 matrix, where D is the dimension of
        the manifold.
        """
        K = np.identity(self.D + 1)
        K[0, 1:] = self.kernel_function_gradient(x1, x2)
        K[1:, 0] = - K[0, 1:]

        P = np.outer(x1 - x2, x1 - x2) / self.l**2
        K[1:, 1:] = (K[1:, 1:] - P) / self.l**2

        return K * self.kernel_function(x1, x2)

    def kernel_matrix(self, X):
        """This is the same method than self.K for X1=X2, but using the matrix
        is then symmetric.
        """
        n, D = np.atleast_2d(X).shape
        K = np.identity(n * (D + 1))
        self.D = D
        D1 = D + 1

        # fill upper triangular:
        for i in range(n):
            for j in range(i + 1, n):
                k = self.kernel(X[i], X[j])
                K[i * D1:(i + 1) * D1, j * D1:(j + 1) * D1] = k
                K[j * D1:(j + 1) * D1, i * D1:(i + 1) * D1] = k.T
            K[i * D1:(i + 1) * D1,
              i * D1:(i + 1) * D1] = self.kernel(X[i], X[i])
        return K

    def kernel_vector(self, x, X, nsample):
        return np.hstack([self.kernel(x, x2) for x2 in X])

    # ---------Derivatives--------
    def dK_dweight(self, X):
        """Return the derivative of K(X,X) respect to the weight """
        return self.K(X, X) * 2 / self.weight

    # ----Derivatives of the kernel function respect to the scale ---
    def dK_dl_k(self, x1, x2):
        """Returns the derivative of the kernel function respect to l"""
        return self.squared_distance(x1, x2) / self.l

    def dK_dl_j(self, x1, x2):
        """Returns the derivative of the gradient of the kernel function
        respect to l
        """
        prefactor = -2 * (1 - 0.5 * self.squared_distance(x1, x2)) / self.l
        return self.kernel_function_gradient(x1, x2) * prefactor

    def dK_dl_h(self, x1, x2):
        """Returns the derivative of the hessian of the kernel function respect
        to l
        """

        Id = np.identity(self.D)
        P = np.outer(x1 - x2, x1 - x2) / self.l**2
        prefactor = 1 - 0.5 * self.squared_distance(x1, x2)
        return -2 * (prefactor * (Id - P) - P) / self.l**3

    def dK_dl_matrix(self, x1, x2):
        k = np.asarray(self.dK_dl_k(x1, x2)).reshape((1, 1))
        j2 = self.dK_dl_j(x1, x2).reshape(1, -1)
        j1 = self.dK_dl_j(x2, x1).reshape(-1, 1)
        h = self.dK_dl_h(x1, x2)
        return np.block([[k, j2], [j1, h]]) * self.kernel_function(x1, x2)

    def dK_dl(self, X):
        """Return the derivative of K(X,X) respect of l"""
        return np.block([[self.dK_dl_matrix(x1, x2) for x2 in X] for x1 in X])

    def gradient(self, X, params_to_update=['weight', 'scale']):
        """
        Computes the gradient of matrix K given the data respect to the
        hyperparameters. Note matrix K here is self.K(X,X).
        Returns a list of n(D+1) x n(D+1) matrices, one for each partial
        derivative
        """
        g = []
        if 'weight' in params_to_update:
            g.append(self.dK_dweight(X))
        if 'scale' in params_to_update:
            g.append(self.dK_dl(X))

        return g


# --------------------------
#      BondedKernel
# --------------------------


class BondExponential(SquaredExponential):
    """
    TODO: Nice documentation here
    """
    def ___init__(self, dimensionality=None):
        # We use the SquaredExponential kernel as a basis
        SquaredExponential.__init__(self)

    # --- Define metric tensor ---

    def init_interaction(self, interaction=None):

        if interaction is None:
            def interaction(x,y):
                return 1.

        if not hasattr(self, 'params'):
            self.params = {}

        for x, y in combinations(self.symbols,2):
            symbols = [x,y]
            symbols.sort()
            param_name = 'f_{}{}'.format(*symbols)
            if param_name in self.params:
                continue
            self.params[param_name] = interaction(x,y)

    def interaction(self, x, y):
        '''Note: x and y must be atomic symbols '''

        try:
            output = self.params['f_%s%s' % (x, y)]
        except KeyError:
            output = self.params['f_%s%s' % (y, x)]

        if self.normalize:
            return output/self.N
        else:
            return output

    def set_metric(self):

        # 1. one-D metric g
        g = np.empty((self.N, self.N))
        d = np.zeros(self.N)

        for i, j in product(range(self.N), range(self.N)):
            if i==j:
                continue
            a_ij =self.interaction(self.symbols[i], self.symbols[j])
            g[i, j] = - a_ij
            d[i] += a_ij
        np.fill_diagonal(g, d)

        # 2. three-D metric G
        self.G = np.zeros((3 * self.N, 3 * self.N))
        self.G[0::3, 0::3] = g[:, :]
        self.G[1::3, 1::3] = g[:, :]
        self.G[2::3, 2::3] = g[:, :]


    def init_metric(self, symbols, interaction=None, normalize=True):

        # Number of atoms
        self.normalize = normalize
        self.N = len(symbols)

        # Save symbol list:
        self.symbols = symbols

        self.init_interaction(interaction)
        self.set_metric()

        
    def set_params(self, params):
        super().set_params(params)
        self.set_metric()

    # --- Kernel methods ---

    def squared_distance(self, x1, x2):
        """ Returns the norm of x1-x2 using G/l**2 as metric tensor """
        return np.sum((x1 - x2) *np.dot(self.G, x1 - x2)) / self.l**2

    def kernel_function_gradient(self, x1, x2):
        """
        Gradient of kernel_function respect to the second entry
        x1 : first data point
        x2 : second data point
        """
        return np.matmul(self.G, (x1 - x2)) / self.l ** 2

    def kernel_function_hessian(self, x1, x2):
        """
        Second derivatives matrix of the kernel function
        """
        u = np.matmul(self.G, (x1 - x2))
        return (self.G - np.outer(u, u) / self.l**2) / self.l**2

    def kernel(self, x1, x2):
        """
        Squared exponential kernel with the matrix G as a metric tensor
        This function returns a D+a x D+1 matrix, where D is the dimension of
        the manifold.
        """

        K = np.identity(self.D + 1)
        K[0, 1:] = self.kernel_function_gradient(x1, x2)
        K[1:, 0] = -K[0, 1:]

        K[1:, 1:] = self.kernel_function_hessian(x1, x2)
        return K * self.kernel_function(x1, x2)

    # --- Kernel derivatives ---

    def dK_dl_h(self, x1, x2):

        """Returns the derivative of the hessian of the kernel function respect
        to l
        """
        u = np.matmul(self.G, x1 - x2)
        P = np.outer(u, u) / self.l**2
        prefactor = 1 - 0.5 * self.squared_distance(x1, x2)
        return -2 * (prefactor * (self.G - P) - P) / self.l**3

    def dG_dfAB(self, A, B):
        g = np.zeros((self.N, self.N))
        d = np.zeros(self.N)

        def connected(C,D):
            if (C,D) == (A,B):
                return 1.
            elif (D,C) == (A,B):
                return 1.
            else:
                return 0.
        
        for i, j in product(range(self.N), range(self.N)):
            if i==j:
                continue
            c = connected(self.symbols[i], self.symbols[j])
            g[i, j] = - c
            d[i] += c
        np.fill_diagonal(g, d)

        # 3-D dG
        dG = np.zeros((3 * self.N, 3 * self.N))
        dG[0::3, 0::3] = g[:, :]
        dG[1::3, 1::3] = g[:, :]
        dG[2::3, 2::3] = g[:, :]

        if self.normalize:
            return dG/self.N
        else:
            return dG


    def dK_dfAB_matrix(self, x1, x2, dG):
        # basic
        tau = x1-x2
        Gtau = np.dot(self.G, tau)
        dGtau = np.dot(dG,tau)
        norm = np.sum(tau * dGtau)/(2*self.l**2)
        # j
        M = dG - norm*self.G
        j = np.dot(M,tau)/self.l**2
        # h
        M2 = np.outer(2*dGtau -norm*Gtau, Gtau)/self.l**2
        h = (M-M2)/self.l**2

        dK = np.empty((self.D+1,self.D+1))
        
        dK[0, 0] = -norm
        dK[0, 1:] = j
        dK[1:, 0] = -j
        dK[1:, 1:] = h
        return dK*self.kernel_function(x1,x2)

    def dK_dfAB(self, X, dG):
        return np.block([[self.dK_dfAB_matrix(x1, x2, dG) for x2 in X] 
                         for x1 in X])

    def gradient(self,X, params_to_update=['weight', 'scale']):
        """
        Computes the gradient of matrix K given the data respect to the
        hyperparameters. Note matrix K here is self.K(X,X).
        Returns a list of n(D+1) x n(D+1) matrices, one for each partial
        derivative
        """
        g = []
        for param in params_to_update:
            if param == 'weight':
                g.append(self.dK_dweight(X))

            elif param == 'scale':
               g.append(self.dK_dl(X))

            elif param.startswith('f_'):
                symbols = re.findall('[A-Z][a-z]?', param[2:])
                dG = self.dG_dfAB(*symbols)
                g.append(self.dK_dfAB(X, dG))
            else:
                raise NameError(f'Parameter name {param} not known')

        return g

