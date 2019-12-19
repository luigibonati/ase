import numpy as np
from scipy.optimize import minimize
from scipy.linalg import solve_triangular, cho_factor, cho_solve
from ase.optimize.activelearning.gp.kernel import SquaredExponential
from ase.optimize.activelearning.gp.prior import ZeroPrior


class GaussianProcess():
    """Gaussian Process Regression
    It is recomended to be used with other Priors and Kernels from
    ase.optimize.gpmin

    Parameters:

    prior: Prior class, as in ase.optimize.gpmin.prior
           Defaults to ZeroPrior

    kernel: Kernel function for the regression, as in
            ase.optimize.gpmin.kernel
            Defaults to the Squared Exponential kernel with derivatives
    """
    def __init__(self, prior=None, kernel=None):
        if kernel is None:
            self.kernel = SquaredExponential()
        else:
            self.kernel = kernel

        if prior is None:
            self.prior = ZeroPrior()
        else:
            self.prior = prior

        self.hyperparams = {}

    def set_hyperparams(self, params, noise):
        """Set hyperparameters of the regression.
        This is a dictionary containing the parameters of the
        kernel as values and their names as keys. 
        The regularization (noise) of the method is specified as
        a separate value. 
        The method defines as weight/noise ratio when it is 
        initiallized, which is subsequently kept during optimization
        of hyperparameters.
        """
        for pstring in params.keys():
            self.hyperparams[pstring] = params.get(pstring)
        self.kernel.set_params(params)
        self.noise = noise
        
        # Set noise-weight ratio:
        if not hasattr(self, 'ratio'):
            self.ratio = self.noise / self.hyperparams['weight']

        # Keep noise-weight ratio as constant
        self.noise = self.ratio * self.hyperparams['weight']

    def train(self, X, Y, noise=None):
        """Produces a PES model from data.

        Given a set of observations, X, Y, compute the K matrix
        of the Kernel given the data (and its cholesky factorization)
        This method should be executed whenever more data is added.

        Parameters:

        X: observations (i.e. positions). numpy array with shape: nsamples x D
        Y: targets (i.e. energy and forces). numpy array with
            shape (nsamples, D+1)
        noise: Noise parameter in the case it needs to be restated.
        """
        if noise is not None:
            self.noise = noise  # Set noise attribute to a different value

        self.X = X.copy()  # Store the data in an attribute
        n = self.X.shape[0]
        D = self.X.shape[1]
        regularization = np.array(n * ([self.noise * self.kernel.l] +
                                  D * [self.noise]))

        K = self.kernel.kernel_matrix(X)  # Compute the kernel matrix
        K[range(K.shape[0]), range(K.shape[0])] += regularization**2

        self.L, self.lower = cho_factor(K, lower=True, check_finite=True)

        # Update the prior if it is allowed to update
        if self.prior.use_update:
            self.prior.update(X, Y, self.L)

        self.m = self.prior.prior(X)

        self.a = Y.flatten() - self.m
        cho_solve((self.L, self.lower), self.a, overwrite_b=True,
                  check_finite=True)

    def predict(self, x, get_variance=False):
        """Given a trained Gaussian Process, it predicts the value and the
        uncertainty at point x. It returns f and V:
        f : prediction: [y, grady]
        V : Covariance matrix. Its diagonal is the variance of each component
            of f.

        Parameters:

        x (1D np.array):      The position at which the prediction is computed
        get_variance (bool):  if False, only the prediction f is returned
                              if True, the prediction f and the variance V are
                              returned: Note V is O(D*nsample2)
        """
        n = self.X.shape[0]
        k = self.kernel.kernel_vector(x, self.X, n)
        f = self.prior.prior(x) + np.dot(k, self.a)
        if get_variance:
            v = solve_triangular(self.L, k.T.copy(), lower=True,
                                 check_finite=False)
            variance = self.kernel.kernel(x, x)
            # covariance = np.matmul(v.T, v)
            covariance = np.tensordot(v, v, axes=(0, 0))
            V = variance - covariance
            return f, V
        return f, None

    def neg_log_likelihood(self, params, *args):
        """
        Negative logarithm of the marginal likelihood and its derivative.
        It has been built in the form that suits the best its optimization,
        with the scipy minimize module, to find the optimal hyperparameters.

        Parameters:

        params: initial value of the hyperparameters to be updated.
        *args: Should be a tuple containing the inputs and targets
               in the training set, and the hyperparameters to update.
        """

        X, Y, params_to_update = args
        
        assert len(params) == len(params_to_update)
    
        paramdict = {}
        for p, pstring in zip(params, params_to_update):
            paramdict[pstring] = p

        self.set_hyperparams(paramdict, self.noise)

        self.train(X, Y)
        y = Y.flatten()

        # Compute log likelihood
        logP = (-0.5 * np.dot(y - self.m, self.a) -
                np.sum(np.log(np.diag(self.L))) -
                X.shape[0] * 0.5 * np.log(2 * np.pi))

        # Gradient of the loglikelihood
        grad = self.kernel.gradient(X,params_to_update)

        # vectorizing the derivative of the log likelihood
        D_P_input = np.array([np.dot(np.outer(self.a, self.a), g)
                              for g in grad])
        D_complexity = np.array([cho_solve((self.L, self.lower), g)
                                 for g in grad])

        DlogP = 0.5 * np.trace(D_P_input - D_complexity, axis1=1, axis2=2)
        return -logP, -DlogP

    def fit_hyperparameters(self, X, Y, params_to_update,
                            bounds=None, tol=1e-2):
        """Given a set of observations, X, Y; optimize the scale
        of the Gaussian Process maximizing the marginal log-likelihood.
        This method calls TRAIN there is no need to call the TRAIN method
        again. The method also sets the parameters of the Kernel to their
        optimal value at the end of execution

        Parameters:

        X:   observations(i.e. positions). numpy array with shape: nsamples x D
        Y:   targets (i.e. energy and forces).
             numpy array with shape (nsamples, D+1)
        params_to_update: list with the names of the hyperparameters to update.
        bounds:  list with the boundaries for the optimization of the 
                 hyperparameters to be updated. Each boundary should be 
                 a tuple (min, max) or None if the optimization should
                 be unconstrained.
        tol: tolerance on the maximum component of the gradient of the
             log-likelihood.
             (See scipy's L-BFGS-B documentation:
             https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html)
        

        Returns:

        result (dict) :
              result = {'hyperparameters': (numpy.array) New hyperparameters,
                        'converged': (bool) True if it converged,
                                            False otherwise
                       }
        """

        assert type(params_to_update[0]) == str

        # Define arguments for neg_log_likelihood
        arguments = (np.array(X), np.array(Y), params_to_update)
        
        # Define initial hyperparameters for minimization
        params = []
        for string in params_to_update:
            params.append(self.hyperparams[string])

        # Optimization
        result = minimize(self.neg_log_likelihood, params, args=arguments,
                          method='L-BFGS-B', jac=True, bounds=bounds,
                          options={'gtol': tol, 'ftol': 0.01 * tol})

        # Collect result
        if not result.success:
            converged = False
        else:
            converged = True
            optimalparams = {}
            for p, pstring in zip(result.x, params_to_update):
                optimalparams[pstring] = p
                self.set_hyperparams(optimalparams, self.noise)

        return {'hyperparameters': self.hyperparams, 'converged': converged}

    def fit_weight_only(self, X, Y, option='update'):
        """
        Fit weight of the kernel keeping all other hyperparameters fixed.
        Here we assume the kernel k(x,x',theta) can be factorized as:
                    k = weight**2 * f(x,x',other hyperparameters)
        this is the case, for example, of the Squared Exponential Kernel.

        Parameters:

        X: observations(i.e. positions). numpy array with shape: nsamples x D
        Y: targets (i.e. energy and forces).
           numpy array with shape (nsamples, D+1)
        option: Whether we just want the value or we want to update the
           hyperparameter. Possible values:
               update: change the weight of the kernel accordingly.
                       Requires a trained Gaussian Process. It
                       works with any kernel.
                       NOTE: the model is RETRAINED

               estimate: return the value of the weight that maximizes
                         the marginal likelihood with all other variables
                         fixed.
                         Requires a trained Gaussian Process with a kernel of
                         value 1.0

        Returns:

        weight: (float) The new weight.
        """

        if not hasattr(self, 'a'):
            self.train(X, Y)

        w = self.hyperparams.get('weight')
        if option == 'estimate':
            assert w == 1.0
        y = Y.flatten()
        m = self.prior.prior(X)
        factor = np.sqrt(np.dot(y - m, self.a) / len(y))

        if option == 'estimate':
            return factor
        elif option == 'update':
            w *= factor
            self.set_hyperparams({'weight': w}, self.noise)
            self.train(X, Y)
            return w
