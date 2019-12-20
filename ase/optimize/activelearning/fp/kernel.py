from __future__ import print_function
import numpy as np
from ase.optimize.activelearning.gp.kernel import SE_kernel

class FPKernel(SE_kernel):

    def __init__(self):
        SE_kernel.__init__(self)

    def set_params(self, params):

        if not hasattr(self, 'params'):
            self.params = {}

        for p in params:
            self.params[p] = params[p]

    def set_fp_params(self, x):

        if self.params != x.params:
            x.update(self.params)

    def kernel_function_gradient(self, x1, x2):
        '''Gradient of kernel_function respect to the second entry.
        x1: first data point
        x2: second data point'''

        gradients = np.array([x1.kernel_gradient(x2, i)
                              for i in range(len(x1.atoms))])
        return gradients.reshape(-1)

    
    def kernel_function_hessian(self, x1, x2):
        d = 3
        hessian = np.zeros([d * len(x1.atoms), d * len(x2.atoms)])

        for i in range(len(x1.atoms)):
            for j in range(len(x2.atoms)):
                hessian[i*d:(i+1)*d, j*d:(j+1)*d] = x1.kernel_hessian(x2, i, j)

        return hessian

    def kernel(self, x1, x2):
        '''Squared exponential kernel including derivatives.
        This function returns a D+1 x D+1 matrix
        where D is the dimension of the manifold'''

        K = np.identity(self.D+1)

        K[0, 0] = x1.kernel(x1, x2)
        K[1:, 0] = self.kernel_function_gradient(x1, x2)
        K[0, 1:] = self.kernel_function_gradient(x2, x1)
        K[1:, 1:] = self.kernel_function_hessian(x1, x2)
        
        return K * self.params.get('weight')**2


    def kernel_matrix(self, X):
        ''' Calculates K(X,X) ie. kernel matrix for training data. '''

        D = len(X[0].atoms) * 3
        n = len(X)
        self.D = D

        # allocate memory
        K = np.identity((n*(D+1)), dtype=float)

        for x in X:
            self.set_fp_params(x)

        for i in range(0, n):
            for j in range(i+1, n):
                k = self.kernel(X[i], X[j])
                K[i*(D+1):(i+1)*(D+1), j*(D+1):(j+1)*(D+1)] = k
                K[j*(D+1):(j+1)*(D+1), i*(D+1):(i+1)*(D+1)] = k.T

            K[i*(D+1):(i+1)*(D+1),
              i*(D+1):(i+1)*(D+1)] = self.kernel(X[i], X[i])


        assert (K == K.T).all()
        return K


    def kernel_vector(self, x, X):
        self.set_fp_params(x)
        for x2 in X:
            self.set_fp_params(x)
        return np.hstack([self.kernel(x, x2) for x2 in X])


        # ---------Derivatives--------
    def dK_dweight(self, X):
        '''Return the derivative of K(X,X) respect to the weight '''

        return self.kernel_matrix(X) * 2 / self.weight

    
    # ----Derivatives of the kernel function respect to the scale ---
    def dK_dl_k(self, x1, x2):
        '''Returns the derivative of the kernel function respect to  l
        '''
        
        return x1.dk_dl(x2)

    
    def dK_dl_j(self, x1, x2):
        '''Returns the derivative of the gradient of the kernel
        function respect to l'''

        # prefactor = (self.distance(x1, x2)**2 / self.l**2 - 2) / self.l
        # return self.kernel_function_gradient(x1, x2) * prefactor

        vector = np.ndarray([len(x1.atoms), 3])
        for atom in x1.atoms:
            vector[atom.index] = x1.d_dl_dk_drm(x2, atom.index)
        return vector.reshape(-1)


    
    def dK_dl_h(self, x1, x2):
        '''Returns the derivative of the hessian of the kernel
        function respect to l'''
        
        d = 3
        matrix = np.ndarray([d * len(x1.atoms), d * len(x2.atoms)])

        for i in range(len(x1.atoms)):
            for j in range(len(x2.atoms)):
                matrix[i*d:(i+1)*d,
                       j*d:(j+1)*d] = x1.d_dl_dk_drm_drn(x2, i, j)

        return matrix

        
    def dK_dl_matrix(self, x1, x2):

        matrix = np.ndarray([self.D+1, self.D+1])

        matrix[0, 0] = self.dK_dl_k(x1, x2)
        matrix[1:, 0] = self.dK_dl_j(x1, x2)
        matrix[0, 1:] = self.dK_dl_j(x2, x1)
        matrix[1:, 1:] = self.dK_dl_h(x1, x2)

        return matrix * self.weight**2

    def dK_dl(self, X):
        '''Return the derivative of K(X,X) respect of l '''

        return np.block([[self.dK_dl_matrix(x1, x2)
                          for x2 in X]
                         for x1 in X])

    
    # Derivatives w.r.t. Delta:
    
    def dK_dDelta_k(self, x1, x2):
        '''Returns the derivative of the kernel function respect to Delta
        '''
        return x1.dk_dDelta(x2)

    
    def dK_dDelta_j(self, x1, x2):
        '''Returns the derivative of the gradient of the kernel
        function respect to Delta'''

        vector = np.zeros([len(x1.atoms), 3])
        for atom in x1.atoms:
            vector[atom.index] = x1.dk_drm_dDelta(x2, atom.index)
        return vector.reshape(-1)


    
    def dK_dDelta_h(self, x1, x2):
        '''Returns the derivative of the hessian of the kernel
        function respect to Delta'''
        
        d = 3
        matrix = np.zeros([d * len(x1.atoms), d * len(x2.atoms)])

        for i in range(len(x1.atoms)):
            for j in range(len(x2.atoms)):
                matrix[i*d:(i+1)*d,
                       j*d:(j+1)*d] = x1.dk_drm_drn_dDelta(x2, i, j)

        return matrix

        
    def dK_dDelta_matrix(self, x1, x2):
        
        matrix = np.ndarray([self.D+1, self.D+1])

        matrix[0, 0] = self.dK_dDelta_k(x1, x2)
        matrix[1:, 0] = self.dK_dDelta_j(x1, x2)
        matrix[0, 1:] = self.dK_dDelta_j(x2, x1)
        matrix[1:, 1:] = self.dK_dDelta_h(x1, x2)

        return matrix * self.weight**2

    def dK_dDelta(self, X):
        '''Return the derivative of K(X,X) respect of l '''

        return np.block([[self.dK_dDelta_matrix(x1, x2)
                          for x2 in X]
                         for x1 in X])
    
    
    def gradient(self, X):
        '''Computes the gradient of matrix K given
        the data respect to the hyperparameters
        Note matrix K here is self.K(X,X)

        returns a 3-entry list of n(D+1) x n(D+1) matrices '''

        for x in X:
            self.set_fp_params(x)
            
        g = [self.dK_dweight(X), self.dK_dl(X), self.dK_dDelta(X)]

        return g
