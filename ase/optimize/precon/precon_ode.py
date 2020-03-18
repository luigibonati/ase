import numpy as np


"""
`ODE12r`: adaptive ODE solver, uses 1st and 2nd order approximations to estimate local error and find a new step length
### Parameters:
* `rtol` : relative tolerance
* `C1` : sufficient contraction parameter
* `C2` : residual growth control (Inf means there is no control)
* `h` : step size, if nothing is passed, an estimate is used based on ODE12
* `hmin` : minimal allowed step size
* `maxF` : terminate if |Fn| > maxF * |F0|
* `extrapolate` : extrapolation style (3 seems the most robust)
"""


def odesolve_r12(f, X0, h=None, verbose=1, fmax=1e-6, maxtol=1e3,
                 steps=100, rtol=1e-1, 
                 C1 = 1e-2, C2=2.0, hmin=1e-10, extrapolate=3, callback=None, precon=None, converged=None):

    X = X0
    
    Fn = f(X) #Get the Forces
    print('Fn', Fn, type(Fn))
    print('element', Fn[2], type(Fn[2]))
    if precon:
        Fn, Rn = precon(Fn, X)
    else:
        Rn = np.linalg.norm(Fn, np.inf)

    
    if Rn <= fmax:
        print(f"ODE12r terminates succesfully after 0 iterations with residual {Rn}")#Forces are already small
        return h

    if Rn >= maxtol:
        print(f"ODE12r: Residual {Rn} is too large at itteration 0") #Forces are too big
        return h

    # computation of the initial step
    r = np.linalg.norm(Fn, np.inf) #pick the biggest force
    if h is None:
        h = 0.5 * rtol**0.5 / r #Chose a stepsize based on that force
        h = max(h, hmin) #Make sure the step size is not too big

    for nit in range(1, steps):

        #Redistribute
        Xnew = X + h * Fn #Pick a new position
        Fnew = f(Xnew) # Calculate the new forces at this position
        if precon:
            Fnew, Rnew = precon(Fnew, Xnew)
        else:
            Rnew = np.linalg.norm(Fnew, np.inf)

        e = 0.5 * h * (Fnew - Fn) #Estimate the area under the foces curve
       
        err = np.linalg.norm(e, np.inf) # Come up with an error based on this area CHECK PAGE 42 OF STELAS THESIS

        #This deceides whether or not to acccept the new residual 
        if Rnew <= Rn*(1-C1*h) or Rnew <= (Rn*C2 and err <= rtol ):
            accept = True

        else:
            accept = False
            conditions = (Rnew <= Rn * (1 - C1 * h), Rnew <= Rn * C2, err <= rtol ) # THIS ALSO SEEMS POTENTIALLY WRONG


        #This decides on an extrapolation scheme for the system, to pick a new increment.
        y = Fn - Fnew
        if extrapolate == 1:       # F(xn + h Fn)
            h_ls = h*np.dot(Fn, y)/(np.dot(y,y))
        elif extrapolate == 2:   # F(Xn + h Fn)
            h_ls = h * np.dot(Fn, Fnew) / (np.dot(Fn, y) + 1e-10)
        elif extrapolate == 3:   # min | F(Xn + h Fn) |
            h_ls = h * np.dot(Fn, y) / (np.dot(y, y) + 1e-10)
        else:
            raise ValueError(f'invalid extrapolate parameter: {extrapolate}')
        if np.isnan(h_ls) or h_ls < hmin: #This rejects the increment if it is too small or the extrapolation scheme misbehaves
            h_ls = np.inf

        # This pickes a separate increment
        h_err = h * 0.5 * np.sqrt(rtol/err)

        #We incremnet the system
        if accept:
            X = Xnew
            Fn = Fnew

            Rn = Rnew
            if callback is not None:
                callback(X)
            

            #We check the residuals again
            if converged is not None:
                conv = converged()
            else:
                conv = Rn <= fmax
            if conv:
                if verbose >= 1:
                    print(f"ODE12r: terminates succesfully after {nit} iterations with residual {Rn} with h equal {h}.")
                return h
            if Rn >= maxtol:
                print(f"ODE12r: Residual {Rn} is too large at iteration number {nit}")
        
                return h

            # Compute a new step size. This is based on the extrapolation and some other heuristics
            h = max(0.25 * h, min(4*h, h_err, h_ls))
            # Log step-size analytic results

        else:
        # Compute a new step size.
            h = max(0.1 * h, min(0.25*h, h_err, h_ls)) #This also computes a new step size if the old one is not allowed 
        # error message if step size is too small
        if abs(h) <= hmin:
            print(f'ODE12r Step size {h} too small at nit = {nit}')
            return h


    #Logging:
    if verbose >= 1:
        print(f'ODE12r terminates unuccesfully after {steps} iterations.')

    return h


#def exp_precon_matrix(X, P)


from ase.optimize.sciopt import SciPyOptimizer, Converged
from ase.optimize.precon import Exp, C1, Pfrommer

class ODE12rOptimizer_precon(SciPyOptimizer):
    def __init__(self, atoms, logfile='-', trajectory=None,
                 callback_always=False, alpha=1.0, master=None,
                 force_consistent=None, precon='Exp',
                 precon_update_tol=1e-3):
        SciPyOptimizer.__init__(self, atoms, logfile, trajectory,
                                callback_always, alpha, master,
                                force_consistent)
        # construct preconditioner if passed as a string
        if isinstance(precon, str):
            if precon == 'C1':
                precon = C1()
            if precon == 'Exp':
                precon = Exp()
            elif precon == 'Pfrommer':
                precon = Pfrommer()
            elif precon == 'ID':
                precon = None
            else:
                raise ValueError('Unknown preconditioner "{0}"'.format(precon))
        self.precon = precon
        self.precon_update_tol = precon_update_tol
        self._last_x = None

    def apply_precon(self, F, x):
        if self._last_x is None:
            # ensure we build precon the first time
            max_move = 2*self.precon_update_tol
        else:
            max_move = np.linalg.norm(x - self._last_x, np.inf)
        if max_move > self.precon_update_tol:
            print(f'Rebuilding preconditioner since max_move={max_move} > tol={self.precon_update_tol}...')
            self.atoms.set_positions(x.reshape((len(self.atoms), 3)))
            self.precon.make_precon(self.atoms)
            self._last_x = x.copy()
        Rp = np.linalg.norm(F, np.inf)            
        Fp = self.precon.solve(F)
        return Fp, Rp

    def run(self, fmax=0.05, steps=100000000):
        if self.force_consistent is None:
            self.set_force_consistent()
        self.fmax = fmax
        try:
            # As SciPy does not log the zeroth iteration, we do that manually
            self.callback(None)    
            # Scale the problem as SciPy uses I as initial Hessian.
            self.call_fmin(fmax / self.H0, steps)
        except Converged:
            pass
         
    def call_fmin(self, fmax, steps):
        precon = None
        if self.precon is not None:
            precon = self.apply_precon
        h = odesolve_r12(lambda x: -self.fprime(x),
                                     self.x0(),
                                     fmax=fmax,
                                     steps=steps,
                                     callback=self.callback,
                                     precon=precon,
                                     converged=self.converged)
