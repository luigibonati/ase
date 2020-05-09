from ase.neb import NEB
import numpy as np


class dyNEB(NEB):
    def __init__(self, images, k=0.1, fmax=0.05, climb=False, parallel=False,
                 remove_rotation_and_translation=False, world=None,
                 dynamic_relaxation=True, scale_fmax=0., method='aseneb'):
        """
        Subclass of NEB that allows for scaled and dynamic optimizations of
        images. This method, which only works in series, does not perform
        force calls on images that are below the convergence criterion.
        The convergence criteria can be scaled with a displacement metric
        to focus the optimization on the saddle point region.

        'Scaled and Dynamic Optimizations of Nudged Elastic Bands',
        P. Lindgren, G. Kastlunger and A. A. Peterson,
        J. Chem. Theory Comput. 15, 11, 5787-5793 (2019).

        dynamic_relaxation: bool
            True skips images with forces below the convergence criterion.
            This is updated after each force call; if a previously converged
            image goes out of tolerance (due to spring adjustments between
            the image and its neighbors), it will be optimized again.

        fmax: float
            Must be identical to the fmax of the optimizer.

        scale_fmax: float
            Scale convergence criteria along band based on the distance between
            an image and the image with the highest potential energy. This
            keyword determines how rapidly the convergence criteria are scaled.
        """
        NEB.__init__(self, images, k=0.1, climb=False, parallel=False,
                     remove_rotation_and_translation=False, world=None,
                     method='aseneb')
        self.fmax = fmax
        self.dynamic_relaxation = dynamic_relaxation
        self.scale_fmax = scale_fmax

        if not self.dynamic_relaxation and self.scale_fmax:
            msg = ('Scaled convergence criteria only implemented in series '
                   'with dynamic relaxation.')
            raise ValueError(msg)

    def set_positions(self, positions):
        n1 = 0
        for i, image in enumerate(self.images[1:-1]):
            if self.dynamic_relaxation:
                if self.parallel:
                    msg = ('Dynamic relaxation does not work efficiently '
                           'when parallelizing over images. Try AutoNEB '
                           'routine for freezing images in parallel.')
                    raise ValueError(msg)
                else:
                    forces_dyn = self._fmax_all(self.images)
                    if forces_dyn[i] < self.fmax:
                        n1 += self.natoms
                    else:
                        n2 = n1 + self.natoms
                        image.set_positions(positions[n1:n2])
                        n1 = n2
            else:
                n2 = n1 + self.natoms
                image.set_positions(positions[n1:n2])
                n1 = n2

    def _fmax_all(self, images):
        '''Store maximum force acting on each image in list. This is used in
           the dynamic optimization routine in the set_positions() function.'''
        n = self.natoms
        f = self.get_forces()
        fmax_images = [np.sqrt((f[n*i:n+n*i]**2).sum(axis=1)).max()
                       for i in range(self.nimages-2)]
        return fmax_images

    def get_forces(self):
        '''Get NEB forces and scale the convergence criteria to focus
           optimization on saddle point region. The keyword scale_fmax
           determines the rate of convergence scaling.'''
        forces = NEB.get_forces(self)
        n = self.natoms
        for i in range(self.nimages-2):
            n1 = n * i
            n2 = n1 + n
            force = np.sqrt((forces[n1:n2]**2.).sum(axis=1)).max()
            n_imax = (self.imax - 1) * n  # Image with highest potential energy

            positions = self.get_positions()
            pos_imax = positions[n_imax:n_imax+n]

            '''Scale convergence criteria based on distance between an image
               and the image with the highest potential energy.'''
            rel_pos = np.sqrt(((positions[n1:n2] - pos_imax)**2).sum())
            if force < self.fmax * (1 + rel_pos * self.scale_fmax):
                if i == self.imax - 1:
                    # Keep forces at saddle point for the log file.
                    pass
                else:
                    # Set forces to zero before they are sent to optimizer.
                    forces[n1:n2, :] = 0
        return forces
