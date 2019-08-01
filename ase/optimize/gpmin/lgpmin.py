from ase.optimize.gpmin.gpmin import GPMin
from ase.optimize.gpmin.gp import GaussianProcess 
from ase.optimize.gpmin.kernel import SE_kernel


import numpy as np
import random 

class LGPMin(GPMin):
    """ Right now it only works if we do not update """
    def __init__(self, atoms, restart=None, logfile='-', trajectory=None, prior=None,
                 master=None, noise=0.005, weight=1., update_prior_strategy='maximum',
                 scale=0.4, force_consistent=None, batch_size=5, bounds = None,
                 update_hyperparams=False, memory = None, use_woodbury = False):


         GPMin.__init__(self, atoms, restart, logfile, trajectory, prior, master,
                        noise, weight, update_prior_strategy, scale, force_consistent, 
                        batch_size, bounds, update_hyperparams)


         if memory is None:
             self.memory = len(self.atoms)//10
         else:
             self.memory = memory

         self.K = None
         self.generation = int(np.around(np.sqrt(self.memory)))
         random.seed(42)

         self.use_woodbury = use_woodbury

    def replace(self,x,y, threshold = None):
        X = np.array(self.x_list)
        A =  self.energy_kernel.K(X,X)

        k = np.array([self.energy_kernel.kernel(x, x2) for x2 in self.x_list])
        c = np.linalg.solve(A + self.noise*self.kernel.l**2, k)

        i = np.argmax(c) #This is the index we use to replace

        if threshold is None or c[i]>threshold:
            if self.use_woodbury:
                self.index_list.append(i)
                k = self.kernel.kernel_vector(x,X, self.memory)
                k -= self.kernel.kernel_vector(self.x_list[i],X,self.memory)
                self.row_list.append(k)

            self.x_list[i] = x
            self.y_list[i] = y
     
            

    def update_training_set(self, x, y):
        

        if len(self.x_list) < self.memory:
             
            self.x_list.append(x)
            self.y_list.append(y)

        else:
           
           if self.K is None:
               
               self.energy_kernel = SE_kernel() 
               self.energy_kernel.set_params(self.hyperparams[:-1])
               self.x_population = self.x_list.copy()
               self.y_population = self.y_list.copy()

           if self.use_woodbury:
               self.index_list = []
               self.row_list = []

           for j in range(self.generation):
               i = random.randrange(len(self.x_population))
               self.replace(self.x_population[i], self.y_population[i], 1.1)

           self.replace(x,y)
           self.x_population.append(x)
           self.y_population.append(y)


    def get_update_matrix(self, i,k, n, D):

        #remember n is the memory
        # add regularization
        S = np.zeros((D+1)*n, D+1)
        #S[(D+1)*i:(D+1)*(i+1), :] = np.diag(np.array(self.noise*self.kernel.l
        #                                  + D*[self.noise]))**2
        k[(D+1)*i:(D+1)*(i+1),:]+= 0.5*np.diag(np.array(self.noise*self.kernel.l
                                              + D*[self.noise]))**2

        # canonical basis
        S[(D+1)*i:(D+1)*(i+1), :] = np.eye(D+1)
        return np.hstack(k,S)           
            


    def train(self, X, Y, noise=None):
        """Overwrite gp.GaussianProcess.train if the Woodbury formula 
        is to be used.
        Warning!!! It cannot be used together with predict with variances."""

        n = X.shape[0]

        if not self.use_woodbury or n < self.memory:
            GaussianProcess.train(self, X, Y, noise = noise)
        else:
            if noise is not None:
                self.noise = noise  # Set noise atribute to a different value

            self.X = X.copy()  # Store the data in an atribute
            D = self.X.shape[1]
            regularization = np.array(n*([self.noise*self.kernel.l]
                                          + D*[self.noise]))
            self.m = self.prior.prior(X)

            if self.memory==n:                    #build K_inv 
                K = self.kernel.kernel_matrix(X)  # Compute the kernel matrix 
                K[range(K.shape[0]), range(K.shape[0])] += regularization**2
                self.K_inv = np.linalg.inv(K)

            else:                                 #update K_inv
                for i,k in zip(self.index_list, self.row_list):
                    U = self.get_update_matrix(i,k,self.memory,D)
                    self.K_inv = symmetric_update(self.K_inv, U)

            a = Y.flatten() - self.m
            self.a = np.dot(self.K_inv, a)


    def update(self, r, e, f):
        """Update the PES:
        update the training set, the prior and the hyperparameters.
        Finally, train the model """

        
        y = np.append(np.array(e).reshape(-1), -1*f.reshape(-1))
        self.update_training_set(r, y)

        
        if self.update_prior:
            if self.strategy == 'average':
                av_e = np.mean(np.array(self.y_list)[:, 0])
                self.prior.set_constant(av_e)
            elif self.strategy == 'maximum':
                max_e = np.max(np.array(self.y_list)[:, 0])
                self.prior.set_constant(max_e)
            elif self.strategy == 'init':
                self.prior.set_constant(e)
                self.update_prior = False

       
        if self.update_hp and self.function_calls % self.nbatch == 0 and self.function_calls != 0:
            self.fit_to_batch()

        
        self.train(np.array(self.x_list), np.array(self.y_list))





def woodbury(P_inv, U, V):
    """
    Woodbury identity for matrix inversion
    (P + UV^T)^-1
    credit: Gregory Gundersen
    """
    k = P_inv.shape[1] 
    tmp = np.einsum('ab,bc,cd->ad',
                    V, P_inv, U, 
                    optimize = ['einsum_path', (1,2),(0,1)])
    B_inv = np.linalg.inv(np.eye(k)+tmp)
    tmp = np.einsum('ab,bc,cd,de,ef->af',
                    P_inv,U, B_inv, V, P_inv, 
                    optimize = ['einsum_path', (0,1),(0,1),(0,2),(0,1)])
    return P_inv - tmp
         
def symmetric_update(P_inv, U):
    """
    Woodbury when U = V (symmetric update)
    """
    k = P_inv.shape[1]
    tmp = np.einsum('ab,bc,cd->ad',
                    U, P_inv, U,
                    optimize = ['einsum_path', (1,2),(0,1)])
    B_inv = np.linalg.inv(np.eye(k)+tmp)
    tmp = np.einsum('ab,bc,cd,de,ef->af',
                    P_inv,U, B_inv, U, P_inv,
                    optimize = ['einsum_path', (0,1),(0,1),(0,2),(0,1)])
    return P_inv - tmp
