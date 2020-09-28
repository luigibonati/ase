import numpy as np

from ase.optimize.sciopt import SciPyOptimizer, OptimizerConvergenceError

"""
Adaptive ODE solver, which uses 1st and 2nd order approximations to
estimate local error and choose a new step length.

This optimizer is described in detail in:

            S. Makri, C. Ortner and J. R. Kermode, J. Chem. Phys.
            150, 094109 (2019)
            https://dx.doi.org/10.1063/1.5064465

Parameters
----------

f : function
    function returning driving force on system
X0 : 1-dimensional array
    initial value of degrees of freedom
rtol : float
    relative tolerance
C1 : float
    sufficient contraction parameter
C2 : float
    residual growth control (Inf means there is no control)
h : float
    step size, if None an estimate is used based on ODE12
hmin : float
    minimal allowed step size
fmax : float
    terminate if `|Fn| > maxF * |F0|`
extrapolate : int
    extrapolation style (3 seems the most robust)
callback : function
    optional callback function to call after each update step
precon: ase.optimize.precon.Precon object
    if present, apply a preconditioner to the optimisation
converged: function
    alternative function to check convergence, rather than
    using a simple norm of the forces.
    
Returns
-------

X: array
    final value of degrees of freedom
"""


def ode12r(f, X0, h=None, verbose=1, fmax=1e-6, maxtol=1e3, steps=100,
           rtol=1e-1, C1=1e-2, C2=2.0, hmin=1e-10, extrapolate=3,
           callback=None, precon=None, converged=None, residual=None):
    X = X0

    Fn = f(X)
    if callback is not None:
        callback(X)
    if precon is not None:
        Fn, Rn = precon(Fn, X)
    elif residual is not None:
        Rn = residual(Fn, X)
    else:
        Rn = np.linalg.norm(Fn, np.inf)

    if Rn <= fmax:
        if verbose:
            print("ODE12r terminates successfully after 0 iterations")
        return X

    if Rn >= maxtol:
        raise OptimizerConvergenceError(f"ODE12r: Residual {Rn} is too large "
                                        "at iteration 0")

    # computation of the initial step
    r = np.linalg.norm(Fn, np.inf)  # pick the biggest force
    if h is None:
        h = 0.5 * rtol ** 0.5 / r  # Chose a stepsize based on that force
        h = max(h, hmin)  # Make sure the step size is not too big

    for nit in range(1, steps):
        Xnew = X + h * Fn  # Pick a new position
        Fnew = f(Xnew)  # Calculate the new forces at this position
        if precon is not None:
            Fnew, Rnew = precon(Fnew, Xnew)
        elif residual is not None:
            Rnew = residual(Fnew, Xnew)
        else:
            Rnew = np.linalg.norm(Fnew, np.inf)

        e = 0.5 * h * (Fnew - Fn)  # Estimate the area under the forces curve
        err = np.linalg.norm(e, np.inf)  # Error estimate

        # Accept step if residual decreases sufficiently and/or error acceptable
        if (Rnew <= Rn * (1 - C1 * h)) or ((Rnew <= Rn * C2) and err <= rtol):
            accept = True
        else:
            accept = False

        # Pick an extrapolation scheme for the system & find new increment
        y = Fn - Fnew
        if extrapolate == 1:  # F(xn + h Fn)
            h_ls = h * np.dot(Fn, y) / (np.dot(y, y))
        elif extrapolate == 2:  # F(Xn + h Fn)
            h_ls = h * np.dot(Fn, Fnew) / (np.dot(Fn, y) + 1e-10)
        elif extrapolate == 3:  # min | F(Xn + h Fn) |
            h_ls = h * np.dot(Fn, y) / (np.dot(y, y) + 1e-10)
        else:
            raise ValueError(f'invalid extrapolate value: {extrapolate}. '
                             'Must be 1, 2 or 3')
        if np.isnan(h_ls) or h_ls < hmin:  # Rejects if increment is too small
            h_ls = np.inf

        h_err = h * 0.5 * np.sqrt(rtol / err)

        # Accept the step and do the update
        if accept:
            X = Xnew
            Fn = Fnew
            Rn = Rnew
            if callback is not None:
                callback(X)

            # We check the residuals again
            if converged is not None:
                conv = converged()
            else:
                conv = Rn <= fmax
            if conv:
                if verbose:
                    print(f"ODE12r: terminates successfully "
                          f"after {nit} iterations.")
                return X
            if Rn >= maxtol:
                print(f"ODE12r: Residual {Rn} is too "
                      f"large at iteration number {nit}")
                return X

            # Compute a new step size.
            # Based on the extrapolation and some other heuristics
            h = max(0.25 * h,
                    min(4 * h, h_err, h_ls))  # Log steep-size analytic results

            if verbose >= 2:
                print(f"ODE12r:      accept: new h = {h}, |F| = {Rn}")
                print(f"ODE12r:                hls = {h_ls}")
                print(f"ODE12r:               herr = {h_err}")
        else:
            # Compute a new step size.
            h = max(0.1 * h, min(0.25 * h, h_err,
                                 h_ls))
            if verbose >= 2:
                print(f"ODE12r:      reject: new h = {h}")
                print(f"ODE12r:               |Fnew| = {Rnew}")
                print(f"ODE12r:               |Fold| = {Rn}")
                print(f"ODE12r:        |Fnew|/|Fold| = {Rnew/Rn}")

        # error message if step size is too small
        if abs(h) <= hmin:
            raise OptimizerConvergenceError('ODE12r terminates unsuccessfully'
                                            f' Step size {h} too small')

    raise OptimizerConvergenceError(f'ODE12r terminates unsuccessfully after '
                                    f'{steps} iterations.')


class ODE12r(SciPyOptimizer):
    def __init__(self, atoms, logfile='-', trajectory=None,
                 callback_always=False, alpha=1.0, master=None,
                 force_consistent=None, precon=None, verbose=0, rtol=1e-2):
        SciPyOptimizer.__init__(self, atoms, logfile, trajectory,
                                callback_always, alpha, master,
                                force_consistent)
        if precon is not None:
            from ase.optimize.precon import make_precon
            self.precon = make_precon(precon)
        else:
            self.precon = None
        self.verbose = verbose
        self.rtol = rtol

    def apply_precon(self, Fn, X):
        if self.precon is None:
            return Fn, np.linalg.norm(Fn, np.inf)
        self.atoms.set_positions(X.reshape(len(self.atoms), 3))
        Fn, Rn = self.precon.apply(Fn, self.atoms)
        return Fn, Rn

    def call_fmin(self, fmax, steps):
        ode12r(lambda x: -self.fprime(x),
               self.x0(),
               fmax=fmax, steps=steps,
               verbose=self.verbose,
               precon=self.apply_precon,
               callback=self.callback,
               converged=self.converged,
               rtol=self.rtol)
