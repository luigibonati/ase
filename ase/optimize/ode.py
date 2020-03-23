import numpy as np

from ase.optimize.sciopt import SciPyOptimizer

"""
Adaptive ODE solver, which uses 1st and 2nd order approximations to 
estimate local error and choose a new step length.

This optimizer is described in detail in:

            S. Makri, C. Ortner and J. R. Kermode, J. Chem. Phys.
            150, 094109 (2019)
            https://dx.doi.org/10.1063/1.5064465

Parameters:

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
"""


def odesolve_r12(f, X0, h=None, verbose=1, fmax=1e-6, maxtol=1e3, steps=100,
                 rtol=1e-1, C1=1e-2, C2=2.0, hmin=1e-10, extrapolate=3,
                 callback=None):
    X = X0
    X_out = []  # Create an array to store the values of X

    Fn = f(X)
    Rn = np.linalg.norm(Fn, np.inf)  # current residual of the forces
    X_out.append(X)
    log = []
    log.append([0, X, Rn])

    if Rn <= fmax:
        print(
            f"ODE12r terminates successfully after 0 iterations")  # Forces are already small
        return X_out, log, h

    if Rn >= maxtol:
        print(
            f"SADDLESEARCH: Residual {Rn} is too large at itteration 0")  # Forces are too big
        return X_out, log, h

    # computation of the initial step
    r = np.linalg.norm(Fn, np.inf)  # pick the biggest force
    if h is None:
        h = 0.5 * rtol ** 0.5 / r  # Chose a stepsize based on that force
        h = max(h, hmin)  # Make sure the step size is not too big

    for nit in range(1, steps):
        # Redistribute
        Xnew = X + h * Fn  # Pick a new position
        Fnew = f(Xnew)  # Calculate the new forces at this position
        Rnew = np.linalg.norm(Fnew, np.inf)  # Find the new residual forces

        e = 0.5 * h * (Fnew - Fn)  # Estimate the area under the foces curve

        err = np.linalg.norm(e,
                             np.inf)  # Come up with an error based on this area

        # This deceides whether or not to acccept the new residual
        if Rnew <= Rn * (1 - C1 * h) or Rnew <= (Rn * C2 and err <= rtol):
            accept = True
        else:
            accept = False
            conditions = (Rnew <= Rn * (1 - C1 * h), Rnew <= Rn * C2,
                          err <= rtol)  # THIS ALSO SEEMS POTENTIALLY WRONG

        # This decides on an extrapolation scheme for the system, to pick a new increment.
        y = Fn - Fnew
        if extrapolate == 1:  # F(xn + h Fn)
            h_ls = h * np.dot(Fn, y) / (np.dot(y, y))
        elif extrapolate == 2:  # F(Xn + h Fn)
            h_ls = h * np.dot(Fn, Fnew) / (np.dot(Fn, y) + 1e-10)
        elif extrapolate == 3:  # min | F(Xn + h Fn) |
            h_ls = h * np.dot(Fn, y) / (np.dot(y, y) + 1e-10)
        else:
            raise ValueError(f'invalid extrapolate parameter: {extrapolate}')
        if np.isnan(
                h_ls) or h_ls < hmin:  # This rejects the increment if it is too small or the extrapolation scheme misbehaves
            h_ls = np.inf

        # This pickes a separate increment
        h_err = h * 0.5 * np.sqrt(rtol / err)

        # We incremnet the system
        if accept:
            X = Xnew
            Fn = Fnew
            Rn = Rnew
            if callback is not None:
                callback(X)

            X_out = np.append(X_out, X)  # Store X
            log = np.append(log, Rn)

            # We check the residuals again
            if Rn <= fmax:
                if verbose >= 1:
                    print(
                        f"SADDLESEARCH: terminates succesfully after {nit} iterations.")
                X_out = np.append(X_out, X)
                log = np.append(log, Rn)
                return X_out, log, h
            if Rn >= maxtol:
                print(
                    f"SADLESEARCH: Residual {Rn} is too large at itteration number {nit}")

                X_out = np.append(X_out, X)  # Store X
                log = np.append(log, Rn)
                return X_out, log, h

            # Compute a new step size. This is based on the extrapolation and some other heuristics
            h = max(0.25 * h,
                    min(4 * h, h_err, h_ls))  # Log step-size analytic results

        else:
            # Compute a new step size.
            h = max(0.1 * h, min(0.25 * h, h_err,
                                 h_ls))  # This also computes a new step size if the old one is not allowed
        # error message if step size is too small
        if abs(h) <= hmin:
            print(f'ODE12r Step size {h} too small at nit = {nit}')
            return X_out, log, h

    # Logging:
    if verbose >= 1:
        print(f'ODE12r terminates unuccesfully after {steps} iterations.')

    return X_out, log, h


class ODE12rOptimizer(SciPyOptimizer):
    def call_fmin(self, fmax, steps):
        X_out, log, h = odesolve_r12(lambda x: -self.fprime(x), self.x0(),
                                     fmax=fmax, steps=steps,
                                     callback=self.callback)
