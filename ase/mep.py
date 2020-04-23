import sys
import time

import numpy as np

from ase.optimize.precon import make_precon
from ase.geometry import find_mic
from ase.optimize.ode import ode12r
from scipy.interpolate import CubicSpline

class MEP:
    def __init__(self, images, k=1.0, precon='Exp', method='NEB',
                 logfile='-'):
        self.images = images
        self.nsteps = 0
        self.natoms = len(images[0])
        for img in images:
            if len(img) != self.natoms:
                raise ValueError('Images have different numbers of atoms')
            if (img.pbc != images[0].pbc).any():
                raise ValueError('Images have different boundary conditions')
            if (img.get_atomic_numbers() !=
                images[0].get_atomic_numbers()).any():
                raise ValueError('Images have atoms in different orders')
        self.nimages = len(images)

        # build an initial preconditioner for each image
        self.precon_method = precon
        self.precon = []
        for i in range(self.nimages):
            P = make_precon(precon)
            P.make_precon(self.images[i])
            self.precon.append(P)

        method = method.lower()
        methods = ['neb', 'string']
        if method not in methods:
            raise ValueError(f'method must be one of {methods}')
        self.method = method

        if isinstance(logfile, str):
            if logfile == "-":
                logfile = sys.stdout
            else:
                logfile = open(logfile, "a")
        self.logfile = logfile

        if isinstance(k, (float, int)):
            k = [k / (self.nimages**2) ] * (self.nimages - 2)

        self.k = list(k)

        self.residuals = np.empty(self.nimages - 2)
        self.fmax_history = []

    def interpolate(self, mic=False):
        """Interpolate the positions of the interior images between the
        initial state (image 0) and final state (image -1).

        mic: bool
            Use the minimum-image convention when interpolating.
        """
        interpolate(self.images, mic)

    def get_positions(self):
        """Get positions as an array of shape ((nimages-2)*natoms, 3)"""
        positions = np.empty(((self.nimages - 2) * self.natoms, 3))
        n1 = 0
        for image in self.images[1:-1]:
            n2 = n1 + self.natoms
            positions[n1:n2] = image.get_positions()
            n1 = n2
        return positions

    def __len__(self):
        """Number of degrees of freedom"""
        return (self.nimages - 2) * self.natoms * 3

    def get_dofs(self):
        """Get degrees of freedom as a long vector"""
        return self.get_positions().reshape(-1)

    def set_positions(self, positions):
        """Set positions from an array of shape ((nimages-2)*natoms, 3)"""
        n1 = 0
        for i, image in enumerate(self.images[1:-1]):
            n2 = n1 + self.natoms
            image.set_positions(positions[n1:n2])
            n1 = n2

    def set_dofs(self, dofs):
        """Set degrees of freedom from a long vector"""
        positions = dofs.reshape(((self.nimages - 2) * self.natoms, 3))
        self.set_positions(positions)

    def spline_fit(self):
        """
        Fit cubic splines to image positions

        Returns
        -------
            s, x_spline
        """

        d_P = np.zeros(self.nimages)
        x = np.zeros((self.nimages, 3 * self.natoms))  # flattened positions
        x[0, :] = self.images[0].positions.reshape(-1)

        for i in range(1, self.nimages):
            x[i] = self.images[i].positions.reshape(-1)
            dx, _ = find_mic(self.images[i].positions -
                             self.images[i - 1].positions,
                             self.images[i - 1].cell,
                             self.images[i - 1].pbc)
            dx = dx.reshape(-1)

            # distance defined in Eq. 8 in the paper
            d_P[i] = np.sqrt(0.5*(self.precon[i].dot(dx, dx) +
                                  self.precon[i - 1].dot(dx, dx)))

        s = d_P.cumsum() / d_P.sum()  # Eq. A1 in the paper
        x_spline = CubicSpline(s, x, bc_type='not-a-knot')
        return s, x_spline

    def get_tangents_and_curvatures(self):
        """Get tangents and curvatures: arrays of shape (nimages-2,natoms*3)"""
        s, x_spline = self.spline_fit(nderivs=2)
        dx_ds = x_spline.derivative()
        d2x_ds2 = x_spline.derivative(2)
        tangents = np.zeros((self.nimages - 2, self.natoms * 3))
        curvatures = np.zeros((self.nimages - 2, self.natoms * 3))
        for i in range(1, self.nimages - 1):
            tangents[i - 1, :] = dx_ds(s[i])
            curvatures[i - 1, :] = d2x_ds2(s[i])
        return tangents, curvatures

    def get_forces(self):
        """Evaluate and return the forces."""
        images = self.images

        calculators = [image.calc for image in images
                       if image.calc is not None]
        if len(set(calculators)) != len(calculators):
            msg = ('One or more NEB images share the same calculator.  '
                   'Each image must have its own calculator.  '
                   'You may wish to use the ase.neb.SingleCalculatorNEB '
                   'class instead, although using separate calculators '
                   'is recommended.')
            raise ValueError(msg)

        forces = np.empty(((self.nimages - 2), self.natoms, 3),
                           dtype=np.float)
        s, x_spline = self.spline_fit()
        dx_ds = x_spline.derivative()
        d2x_ds2 = x_spline.derivative(2)

        self.residuals[:] = 0

        # Evaluate forces for all images - one at a time
        for i in range(1, self.nimages - 1):
            f = images[i].get_forces()

            # update preconditioners for each image and apply to forces
            # this implements part of Eq. 6: pf = - P^{-1} * \nabla V(x)
            pf, _ = self.precon[i].apply(f, images[i])
            pf_vec = pf.reshape(-1)

            # Project out the component parallel to band following Eqs. 6 and 7

            t_P = dx_ds(s[i])
            t_P /= self.precon[i].norm(t_P)
            t_P_tensor_t_P = np.outer(t_P, t_P)
            pf_vec -= t_P_tensor_t_P @ pf_vec

            # print('norm(pf_vec, inf)', np.linalg.norm(pf_vec, np.inf))

            # Definition of residuals on each image from Eq. 11
            self.residuals[i - 1] = np.linalg.norm(self.precon[i].Pdot(pf_vec),
                                                   np.inf)

            if self.method == 'neb':
                # Definition following Eq. 9
                eta_Pn = self.k[i - 1] * self.precon[i].dot(d2x_ds2(s[i]), t_P) * t_P

                # complete Eq. 9 by including the spring force
                pf_vec += eta_Pn

            # print('norm(pf_vec, inf)', np.linalg.norm(pf_vec, np.inf))

            forces[i - 1] = pf_vec.reshape((self.natoms, 3))

        return forces

    def iterimages(self):
        for atoms in self.images:
            yield atoms

    def get_residual(self, F=None, X=None):
        return np.max(self.residuals) # Eq. 11

    def driving_force(self, X):
        self.set_dofs(X)
        f = self.get_forces()
        return f.reshape(-1)

    def log(self):
        fmax = self.get_residual()
        self.fmax_history.append(fmax)
        T = time.localtime()
        if self.logfile is not None:
            name = f'{self.__class__.__name__}[{self.method},{self.optimizer},{self.precon_method}]'
            if self.nsteps == 0:
                args = (
                " " * len(name), "Step", "Time", "fmax")
                msg = "%s  %4s %8s %12s\n" % args
                self.logfile.write(msg)

            args = (name, self.nsteps, T[3], T[4], T[5], fmax)
            msg = "%s:  %3d %02d:%02d:%02d %12.4f\n" % args
            self.logfile.write(msg)
            self.logfile.flush()

    def callback(self, X):
        self.log()
        self.nsteps += 1

        if self.method == 'string':
            # for string we need to reparameterise after each update step
            self.set_dofs(X)
            s, x_spline = self.spline_fit()
            new_s = np.linspace(0, 1, self.nimages)
            X[:] = x_spline(new_s[1:-1]).reshape(-1)


    def run(self, fmax=1e-3, steps=50, optimizer='ODE', alpha=0.01, rtol=0.1):
        optimizer = optimizer.lower()
        optimizers = ['ode', 'static']
        if optimizer not in optimizers:
            raise ValueError(f'optimizer must be one of {optimizers}')
        self.optimizer = optimizer # save for logging purposes

        if optimizer == 'ode':
            ode12r(self.driving_force,
                   self.get_dofs(),
                   fmax=fmax,
                   rtol=rtol,
                   steps=steps,
                   callback=self.callback,
                   residual=self.get_residual)
        else:
            X = self.get_dofs()
            for step in range(steps):
                F = self.driving_force(X)
                if self.get_residual() <= fmax:
                    break
                X += alpha * F
                self.callback(X)

