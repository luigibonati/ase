from ase.optimize.gpmin.gpmin import GPMin 
from ase.optimize.gpmin.kernel import SE_kernel, SquaredExponential
from ase.optimize.gpmin.prior import ConstantPrior

import numpy as np
import random 

class LGPMin(GPMin):
    """ Right now it only works if we do not update """
    def __init__(self, atoms, restart=None, logfile='-', trajectory=None, prior=None,
                 master=None, noise=0.005, weight=1., update_prior_strategy='maximum',
                 scale=0.4, force_consistent=None, batch_size=5, bounds = None,
                 update_hyperparams=False, memory = None):


         GPMin.__init__(self, atoms, restart, logfile, trajectory, prior, master,
                        noise, weight, update_prior_strategy, scale, force_consistent, 
                        batch_size, bounds, update_hyperparams)


         if memory is None:
             self.memory = len(self.atoms)//10
         else:
             self.memory = memory

         self.K = None
         self.generation = np.around(np.sqrt(self.memory))
         random.seed(42)

    def replace(self,x,y, threshold = None):
        X = np.array(self.x_list)
        A =  self.K.K(X,X)

        k = np.array([self.K.kernel(x, x2) for x2 in self.x_list])
        c = np.linalg.solve(A + self.noise*self.kernel.l**2, k)

        i = np.argmax(c) #This is the index we use to replace

        if threshold is None or c[i]>threshold:
            
            self.x_list[i] = x
            self.y_list[i] = y

    def update_training_set(self, x, y):
 
        if len(self.x_list) < self.memory:
             
            self.x_list.append(x)
            self.y_list.append(y)

        else:
           
           if self.K is None:
               
               self.K = SE_kernel() 
               self.K.set_params(self.hyperparams[:-1])
               self.x_population = self.x_list.copy()
               self.y_population = self.y_list.copy()
           
           for j in range(self.generation):
               i = random.randrange(len(self.x_population))
               self.replace(self.x_population[i], self.y_population[i], 1.1)

           self.replace(x,y)
           self.x_population.append(x)
           self.y_population.append(y)


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


