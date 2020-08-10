from ase.optimize.activelearning.gpfp.kernel import SE_kernel, FPKernel

import numpy as np

from scipy.optimize import minimize
from scipy.linalg import solve_triangular, cho_factor, cho_solve

from ase.optimize.activelearning.gpfp.prior import ZeroPrior


class GaussianProcess():

    '''Gaussian Process Regression
    It is recomended to be used with other Priors and Kernels
    from ase.optimize.gpmin

    Parameters:

    prior: Prior class, as in ase.optimize.gpmin.prior
        Defaults to ZeroPrior

    kernel: Kernel function for the regression, as
       in ase.optimize.gpmin.kernel
        Defaults to the Squared Exponential kernel with derivatives '''

    def __init__(self, prior=None, kernel=None,
                 use_forces=True, train_delta=False,
                 noisefactor=0.5):

        self.use_forces = use_forces
        self.train_delta = train_delta

        if kernel is None:
            if self.use_forces:
                self.kernel = FPKernel()
            else:
                self.kernel = SE_kernel()
        else:
            self.kernel = kernel

        if prior is None:
            self.prior = ZeroPrior()
        else:
            self.prior = prior

        self.hyperparams = {}

        self.noisefactor = noisefactor

    def set_hyperparams(self, params, noise):
        '''Set hyperparameters of the regression.
        This is a list containing the parameters of the
        kernel and the regularization (noise)
        of the method as the last entry. '''

        for pstring in params.keys():
            self.hyperparams[pstring] = params.get(pstring)

        self.kernel.set_params(params)
        self.noise = noise

        # Set noise-weight ratio:
        if not hasattr(self, 'ratio'):
            self.ratio = self.noise / self.hyperparams['weight']

        # Keep noise-weight ratio as constant:
        self.noise = self.ratio * self.hyperparams['weight']

    def train(self, X, Y, noise=None):
        '''Produces a PES model from data.

        Given a set of observations, X, Y, compute the K matrix
        of the Kernel given the data (and its cholesky factorization)
        This method should be executed whenever more data is added.

        Parameters:

        X: observations(i.e. positions). numpy array with shape: nsamples x D
        Y: targets (i.e. energy and forces). numpy array with
            shape (nsamples, D+1)
        noise: Noise parameter in the case it needs to be restated.'''

        X = np.array(X)
        Y = np.array(Y)

        if noise is not None:
            self.noise = noise  # Set noise attribute to a different value

        K = self.kernel.kernel_matrix(X)  # Compute the kernel matrix

        self.X = X  # Store the data in an attribute

        n = len(self.X)  # number of training points

        if self.use_forces:
            D = len(self.X[0].atoms) * 3  # number of derivatives
            regularization = np.array(n * ([self.noise *
                                            self.noisefactor] +
                                           D * [self.noise]))

        else:
            regularization = np.array(n * ([self.noise *
                                            self.noisefactor]))

        K += np.diag(regularization**2)
        self.K = K  # this is only used in self.leaveoneout()

        self.L, self.lower = cho_factor(K, lower=True, check_finite=True)

        # Update the prior if it is allowed to update
        if self.prior.use_update:
            self.prior.update(X, Y, self.L)

        Y = Y.flatten()
        # self.m = list(self.prior.prior(np.ones(D))) * n
        self.m = list(np.hstack([self.prior.prior(x) for x in X]))

        assert len(Y) == len(self.m)
        self.a = Y - self.m

        cho_solve((self.L, self.lower), self.a,
                  overwrite_b=True, check_finite=True)

    def predict(self, x, get_variance=False):
        '''Given a trained Gaussian Process, it predicts the value and the
        uncertainty at point x.
        It returns f and V:
        f : prediction: [y, grady]
        V : Covariance matrix. Its diagonal is the variance of each component of f.

        Parameters:

        x (1D np.array): The position at which the prediction is computed
        get_variance (bool): if False, only the prediction f is returned
                            if True, the prediction f and the variance V are
                            returned: Note V is O(D*nsample2)'''

        k = self.kernel.kernel_vector(x, self.X)

        priorarray = self.prior.prior(x)
        f = priorarray + np.dot(k, self.a)

        if get_variance:
            v = k.T.copy()
            v = solve_triangular(self.L, v, lower=True, check_finite=False)

            variance = self.kernel.kernel(x, x)
            covariance = np.tensordot(v, v, axes=(0, 0))

            V = variance - covariance

            return f, V
        return f, None

    def neg_log_likelihood(self, params, *args, fit_weight=True):
        '''Negative logarithm of the marginal likelihood and its derivative.
        It has been built in the form that suits the best its optimization,
        with the scipy minimize module, to find the optimal hyperparameters.

        Parameters:

        l: The scale for which we compute the marginal likelihood
        *args: Should be a tuple containing the inputs and targets
               in the training set- '''

        X, Y, params_to_update = args

        assert len(params) == len(params_to_update)

        params = np.power(10, params)

        txt1 = ""
        paramdict = {}
        for p, pstring in zip(params, params_to_update):
            paramdict[pstring] = p
            txt1 += "{:18.06f}".format(p)

        self.set_hyperparams(paramdict, self.noise)
        self.train(X, Y)

        if fit_weight:
            self.fit_weight_only(X, Y, option='update')

        y = Y.flatten()
        # Compute log likelihood
        logP = (-0.5 * np.dot(y - self.m, self.a)
                - np.sum(np.log(np.diag(self.L)))
                - len(y) / 2 * np.log(2 * np.pi))

        # Don't let ratio fall too small, resulting in numerical
        # difficulties:
        if 'ratio' in params_to_update:
            ratio = params[params_to_update.index('ratio')]
            if ratio < 1e-6:
                logP -= (1e-6 - ratio) * 1e6

        # # Gradient of the loglikelihood
        # grad = self.kernel.gradient(X)

        # # vectorizing the derivative of the log likelyhood
        # D_P_input = np.array([np.dot(np.outer(self.a, self.a), g) for g in grad])
        # D_complexity = np.array([cho_solve((self.L, self.lower), g)
        #                          for g in grad])

        # DlogP = 0.5 * np.trace(D_P_input - D_complexity, axis1=1, axis2=2)
        # txt2 = ""
        # for value in DlogP:
        #     txt2 += "%12.03f" % (-value)
        # print("Parameters:", txt1, "       -logP: %12.02f       -DlogP: " % -logP, txt2)
        # # print("Parameters:", txt1, "       -logP: %12.02f" % -logP)
        # return -logP , -DlogP

        print("Parameters: {:s}       -logP: {:12.02f}".format(txt1, -logP))
        return -logP

    def fit_hyperparameters(self, X, Y,
                            params_to_update,
                            bounds=None, tol=1e-2):
        '''Given a set of observations, X, Y; optimize the scale
        of the Gaussian Process maximizing the marginal log-likelihood.
        This method calls TRAIN there is no need to call the TRAIN method again.
        The method also sets the parameters of the Kernel to their optimal value at
        the end of execution

        Parameters:

        X: observations(i.e. positions). numpy array with shape: nsamples x D
        Y: targets (i.e. energy and forces).
           numpy array with shape (nsamples, D+1)
        tol: tolerance on the maximum component of the gradient of the log-likelihood.
           (See scipy's L-BFGS-B documentation:
           https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html )
        eps: include bounds to the hyperparameters as a +- a percentage of hyperparameter
            if eps is None, there are no bounds in the optimization

        Returns:

        result (dict) :
              result = {'hyperparameters': (numpy.array) New hyperparameters,
                        'converged': (bool) True if it converged,
                                            False otherwise
                       }


        '''

        assert type(params_to_update[0]) == str

        X = np.array(X)
        Y = np.array(Y)

        arguments = (X, Y, params_to_update)

        import time
        t0 = time.time()

        params = []
        for string in params_to_update:
            params.append(np.log10(self.hyperparams[string]))

        bounds = np.log10(bounds)

        result = minimize(self.neg_log_likelihood,
                          params,
                          args=arguments,
                          method='L-BFGS-B',
                          # jac=True,
                          bounds=bounds,
                          options={'gtol': tol, 'ftol': 0.01 * tol})

        print("Time spent minimizing neg log likelihood: %.02f sec" %
              (time.time() - t0))

        if not result.success:
            converged = False

        else:
            converged = True

        # collect results:
        optimalparams = {}
        powered_results = np.power(10, result.x)
        for p, pstring in zip(powered_results, params_to_update):
            optimalparams[pstring] = p

        if 'ratio' in params_to_update:
            self.ratio = optimalparams.get('ratio')

        self.set_hyperparams(optimalparams, self.noise)
        print(self.hyperparams, converged)
        return {'hyperparameters': self.hyperparams, 'converged': converged}

    def fit_weight_only(self, X, Y, option='update'):
        """Fit weight of the kernel keeping all other hyperparameters fixed.
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

        y = np.array(Y).flatten()
        m = list(np.hstack([self.prior.prior(x) for x in X]))
        factor = np.sqrt(np.dot(y - m, self.a) / len(y))

        if option == 'estimate':
            return factor
        elif option == 'update':
            new_weight = factor * w
            self.set_hyperparams({'weight': new_weight}, self.noise)

            # Rescale accordingly ("re-train"):
            self.a /= factor**2
            self.L *= factor
            self.K *= factor**2

            return factor

    def leaveoneout(self, X, Y, index):
        """ Split K-matrix so that the rows and columns that correspond
        to index 'index' are removed, then regularize, then invert the
        matrix, and calculate and return predicted energy and uncertainty.

        Parameters:

        X: Full training point vector

        Y: Full training data vector

        index: index to be removed from full K-matrix

        """

        if self.use_forces:
            d = 1 + len(X[0].atoms) * 3
        else:
            d = 1

        start = index * d
        end = (index + 1) * d

        K = self.K.copy()
        K = np.delete(K, range(start, end), axis=0)
        K = np.delete(K, range(start, end), axis=1)

        Xi = X[index]
        Xreduced = np.concatenate((X[:index], X[index + 1:]))
        Yreduced = np.concatenate((Y[:index], Y[index + 1:]))

        # Cholesky factorization:
        L, lower = cho_factor(K, lower=True, check_finite=True)

        Yreduced = np.array(Yreduced)
        Yreduced = Yreduced.flatten()

        # Calculate prior:
        m = list(np.hstack([self.prior.prior(x) for x in Xreduced]))

        # Test dimensions:
        assert len(Yreduced) == len(m)

        # Calculate alpha:
        a = Yreduced - m

        #  Update self.a with Cholesky solution:
        cho_solve((L, lower), a, overwrite_b=True, check_finite=True)

        # Predict energy and variance for the left-out data point:
        x = Xi
        k = self.kernel.kernel_vector(x, Xreduced)
        priorarray = self.prior.prior(x)
        f = priorarray + np.dot(k, a)

        # Get variance:
        v = k.T.copy()
        v = solve_triangular(L, v, lower=True, check_finite=False)
        variance = self.kernel.kernel(x, x)
        covariance = np.tensordot(v, v, axes=(0, 0))
        V = variance - covariance

        return f, V
