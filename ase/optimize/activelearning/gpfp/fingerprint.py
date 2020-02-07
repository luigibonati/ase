from scipy.spatial.distance import pdist
from math import pi
from scipy.spatial import distance_matrix
import numpy as np
from numpy import dot
from warnings import catch_warnings, simplefilter


class RadialAngularFP():

    def __init__(self, limit=10.0, Rlimit=4.0,
                 delta=0.2, ascale=0.2,
                 N=200, Na=100, aweight=1.0,
                 pbc=None, calc_gradients=True,
                 weight=1.0, scale=1.0):
        ''' Parameters:

        limit: float
               Threshold for radial fingerprint (Angstroms)

        Rlimit: float
                Threshold for angular fingerprint (Angstroms)

        delta: float
               Width of Gaussian broadening in radial fingerprint
               (Angstroms)

        ascale: float
                Width of Gaussian broadening in angular fingerprint
               (Radians)

        N: int
           Number of bins in radial fingerprint

        Na: int
            Number of bins in angular fingerprint

        aweight: float
            Scaling factor for the angular fingerprint; the angular
            fingerprint is multiplied by this number

        pbc: bool, list, None
             Choose whether periodic boundary conditions are
             considered.
             True: Periodic in all directions.
             False: Non-periodic in all directions.
             list of type [bool, bool, bool]: Indicating periodicity
                                              to x,y,z directions
             None: Periodicity information is inherited from the
                   atoms object attached to 'self'

        calc_gradients: bool
                        Whether gradients are calculated

        weight: float
                Sqrt of the prefactor for squared-exponential kernel
                TODO: this should be removed from fingerprint...

        scale: float
               Scale for squared-exponential kernel
               TODO: this should be removed from fingerprint...

        TODO: Rename class attributes.

        '''

        self.limit = limit
        self.delta = delta
        self.N = N
        self.pbc = pbc
        self.nanglebins = Na
        self.Rtheta = Rlimit  # angstroms

        assert self.limit >= self.Rtheta

        self.aweight = aweight
        self.ascale = ascale  # rads

        self.gamma = 2

        # modifiable parameters:
        self.params = {'weight': weight,
                       'scale': scale,
                       'delta': self.delta,
                       'ascale': self.ascale,
                       'aweight': self.aweight}

        self.calc_gradients = calc_gradients
        self.weight_by_elements = True
        self.set_params()

    def set_params(self):
        ''' Set parameters according to dictionary
            self.params '''

        self.weight = self.params.get('weight')
        self.l = self.params.get('scale')
        self.delta = self.params.get('delta')
        self.ascale = self.params.get('ascale')

        return

    def set_atoms(self, atoms):
        ''' Set new atoms and initialize '''

        self.atoms = atoms
        self.Vuc = self.atoms.get_volume()

        if self.pbc is None:
            self.pbc = self.atoms.pbc
        elif self.pbc is False:
            self.pbc = np.array([False, False, False])
        elif self.pbc is True:
            self.pbc = np.array([True, True, True])

        self.elements = np.sort(list(set([atom.symbol for atom in atoms])))
        self.elcounts = [len([atom for atom in self.atoms if
                             atom.symbol == self.elements[i]])
                         for i in range(len(self.elements))]

        self.n = len(self.elements)

        self.extend_positions()
        self.set_angles()
        self.update()

    def update(self, params=None):
        ''' Update method when parameters are changed '''

        if params is not None:
            for param in params:

                self.params[param] = params[param]

        self.set_params()

        self.set_peak_heights()
        self.get_fingerprint()

        if self.calc_gradients:
            self.calculate_all_gradients()

        self.get_angle_fingerprint()

        if self.calc_gradients:
            self.calculate_all_angle_gradients()

        self.dFP_dDelta_calculated = False
        self.dGij_dDelta_calculated = False
        self.d_dDelta_dFP_drm_calculated = False

    def extend_positions(self):
        ''' Extend the unit cell so that all the atoms within the limit
        are in the same cell, indexed properly.
        '''

        # Determine unit cell parameters:
        cell = self.atoms.cell.array
        lengths = self.atoms.cell.lengths()
        natoms = len(self.atoms)

        self.origcell = cell

        # Number of cells needed to consider given the limit and pbc:
        ncells = [self.limit // lengths[i] + 1 for i in range(3)]
        nx, ny, nz = [1 + 2 * int(n) * self.pbc[i]
                      for i, n in enumerate(ncells)]

        self.extendedatoms = self.atoms.repeat([nx, ny, nz])

        newstart = natoms * int(np.prod([nx, ny, nz]) / 2)
        newend = newstart + natoms
        self.atoms = self.extendedatoms[newstart:newend]

        # Distance matrix
        self.dm = distance_matrix(x=self.atoms.positions,
                                  y=self.extendedatoms.positions)

        # position vector matrix
        self.rm = np.einsum('ilkl->ikl',
                            np.subtract.outer(self.atoms.positions,
                                              self.extendedatoms.positions))

        return

    def set_angles(self):
        """
        In angle vector 'self.av' all angles are saved where
        one of the atoms is in 'self.atoms' and the other
        two are in 'self.extendedatoms'
        """

        # Extended distance and displacement vector matrices:
        self.edm = distance_matrix(x=self.extendedatoms.positions,
                                   y=self.extendedatoms.positions)

        ep = self.extendedatoms.positions
        self.erm = np.einsum('ilkl->ikl', np.subtract.outer(ep, ep))

        fcij = self.cutoff_function(self.dm)
        fcjk = self.cutoff_function(self.edm)
        self.angleconstant = self.aweight / (pi / self.nanglebins)

        # angle vector
        self.av = []

        mask1 = np.logical_or(self.dm == 0, self.dm > self.Rtheta)
        mask2 = self.dm == 0
        mask3 = np.logical_or(self.edm == 0, self.edm > self.Rtheta)

        for i in range(len(self.atoms)):
            for j in range(len(self.extendedatoms)):

                if mask1[i, j]:
                    continue

                for k in range(len(self.extendedatoms)):

                    if mask2[i, k]:
                        continue

                    if mask3[j, k]:
                        continue

                    # Argument for arccos:
                    argument = (dot(self.rm[i, j], self.erm[k, j])
                                / self.dm[i, j] / self.edm[k, j])

                    # Handle numerical errors in perfect lattices:
                    if argument >= 1.0:
                        argument = 1.0 - 1e-9
                    elif argument <= -1.0:
                        argument = -1.0 + 1e-9

                    self.av.append([i, j, k, fcij[i, j],
                                    fcjk[k, j], np.arccos(argument)])

        return self.av

    def set_peak_heights(self):
        ''' Calculate the delta peak heights self.h '''

        self.constant = 1 / (self.limit / self.N)
        # print("Constant: ", self.constant)

        # Ignore zero-division warning
        with catch_warnings():
            simplefilter("ignore", category=RuntimeWarning)
            self.h = np.where(self.dm > 0.0001,
                              (self.constant / (self.dm**2)),
                              0.0)

        return self.h

    def cutoff_function(self, r):
        """
        Rtheta: cutoff radius, given in angstroms
        """

        return np.where(r <= self.Rtheta,
                        (1 + self.gamma * (r / self.Rtheta)**(self.gamma+1) -
                         (self.gamma + 1) * (r / self.Rtheta)**self.gamma),
                        0.0)

    def get_fingerprint(self):
        ''' Calculate the Gaussian-broadened fingerprint. '''

        self.G = np.ndarray([self.n, self.n, self.N])
        x = np.linspace(0, self.limit, self.N)  # variable array

        # Broadening of each peak:
        for i in range(self.n):
            for j in range(self.n):

                # Get peak positions
                R = np.where(self.dm < self.limit,
                             self.dm,
                             0.0)

                # Get peak heights
                h = np.where(self.dm < self.limit,
                             self.h,
                             0.0)

                # Consider only the correct elements i and j
                ms = (np.array(self.atoms.get_chemical_symbols()) ==
                      self.elements[i])
                ns = (np.array(self.extendedatoms.get_chemical_symbols()) ==
                      self.elements[j])

                R = (R.T * ms).T
                R = R * ns
                R = R.flatten()

                h = (h.T * ms).T
                h = h * ns
                h = h.flatten()

                # Remove zero-height peaks
                nz = np.nonzero(h)
                R = R[nz]
                h = h[nz]

                npeaks = len(R)  # number of peaks

                g = np.zeros(self.N)
                for p in range(npeaks):
                    g += h[p] * np.exp(- (x - R[p])**2 / 2 / self.delta**2)

                self.G[i, j] = g

        if self.weight_by_elements:
            factortable = np.einsum('i,j->ij',
                                    self.elcounts,
                                    self.elcounts).astype(float)**-1
            self.G = np.einsum('ijk,ij->ijk', self.G, factortable)

        return self.G

    def get_angle_fingerprint(self):
        ''' Calculate the angular fingerprint with Gaussian broadening  '''

        self.H = np.zeros([self.n, self.n, self.n, self.nanglebins])
        x = np.linspace(0, pi, self.nanglebins)  # variable array

        elementlist = list(self.elements)
        isymbols = [elementlist.index(atom.symbol)
                    for atom in self.atoms]
        jsymbols = [elementlist.index(atom.symbol)
                    for atom in self.extendedatoms]
        ksymbols = jsymbols

        # Broadening of each peak:
        for data in self.av:
            i, j, k, fcij, fcjk, theta = data

            A = isymbols[i]
            B = jsymbols[j]
            C = ksymbols[k]

            self.H[A, B, C] += (fcij * fcjk *
                                np.exp(- (x - theta)**2 / 2 / self.ascale**2))

        self.H *= self.angleconstant

        if self.weight_by_elements:
            factortable = np.einsum('i,j,k->ijk',
                                    self.elcounts,
                                    self.elcounts,
                                    self.elcounts).astype(float)**-1
            self.H = np.einsum('ijkl,ijk->ijkl', self.H, factortable)

        return self.H

    def get_fingerprint_vector(self):
        ''' Return the full fingerprint vector. '''
        
        return np.concatenate((self.G.flatten(), self.H.flatten()), axis=None)

    # ::: GRADIENTS ::: #
    # ----------------- #

    def calculate_gradient(self, index):
        '''
        Calculates the derivative of the fingerprint
        with respect to one of the coordinates.

        index: Atom index with which to differentiate
        '''
        gradient = np.zeros([self.n, self.N, 3])

        i = index
        A = list(self.elements).index(self.atoms[i].symbol)

        elementlist = list(self.elements)
        jsymbols = [elementlist.index(atom.symbol)
                    for atom in self.extendedatoms]
        # Sum over elements:
        for B in range(self.n):

            jsum = np.zeros([self.N, 3])

            # sum over atoms in extended cell
            for j in range(len(self.extendedatoms)):

                # atom j is of element B:
                if B != jsymbols[j]:
                    continue

                # position vector between atoms:
                rij = self.rm[i, j]
                Gij = self.Gij(i, j)
                jsum += np.outer(Gij, -rij)

            gradient[B] = ((1 + int(A == B)) * jsum)

        if self.weight_by_elements:
            factortable = (self.elcounts[A] * 
                           np.array(self.elcounts)).astype(float)**-1
            gradient = [gradient[i] * factortable[i]
                        for i in range(len(gradient))]

        return gradient

    def calculate_all_gradients(self):
        ''' Calculate all gradients for radial fingerprint '''
        self.gradients = np.array([self.calculate_gradient(atom.index)
                                   for atom in self.atoms])
        return self.gradients

    # ANGLES:

    def nabla_fcij(self, m, n):
        d = self.dm[m, n]
        r = self.rm[m, n]
        dfc_dd = (self.gamma * (self.gamma + 1) / self.Rtheta *
                  ((d / self.Rtheta) ** self.gamma -
                   (d / self.Rtheta) ** (self.gamma - 1)))
        dd_drm = r / d
        return dfc_dd * dd_drm

    def nabla_fcjk(self, m, n):
        d = self.edm[m, n]
        r = self.erm[m, n]
        dfc_dd = (self.gamma * (self.gamma + 1) / self.Rtheta *
                  ((d / self.Rtheta) ** self.gamma -
                   (d / self.Rtheta) ** (self.gamma - 1)))
        dd_drm = r / d
        return dfc_dd * dd_drm

    def dthetaijk_dri(self, i, j, k, theta):
        r1 = self.dm[i, j]
        v1 = self.rm[i, j]

        r2 = self.edm[k, j]
        v2 = self.erm[k, j]

        dotp = dot(v1, v2)

        if theta == 0.0:
            print("theta=0")
        if r1 == 0.0:
            print("r1=0")
        if r2 == 0.0:
            print("r2=0")

        return 1 / abs(np.sin(theta)) / r1 / r2 * (dotp / r1**2 * v1 - v2)

    def dthetaijk_drj(self, i, j, k, theta):
        r1 = self.dm[i, j]
        v1 = self.rm[i, j]

        r2 = self.edm[k, j]
        v2 = self.erm[k, j]

        dotp = dot(v1, v2)

        prefactor = -1 / r1 / r2 / abs(np.sin(theta))
        first = (-1 + dotp / r1**2) * v1
        second = (-1 + dotp / r2**2) * v2

        return prefactor * (first + second)

    def dthetaijk_drk(self, i, j, k, theta):
        r1 = self.dm[i, j]
        v1 = self.rm[i, j]

        r2 = self.edm[k, j]
        v2 = self.erm[k, j]

        dotp = dot(v1, v2)

        return -1 / abs(np.sin(theta)) / r1 / r2 * (v1 - dotp / r2**2 * v2)

    def calculate_angle_gradient(self, index):
        '''
        Calculates the derivative of the fingerprint
        with respect to one of the coordinates.

        index: Atom index with which to differentiate
        '''
        gradient = np.zeros([self.n, self.n, self.n, self.nanglebins, 3])
        xvec = np.linspace(0., pi, self.nanglebins)

        elementlist = list(self.elements)
        isymbols = [elementlist.index(atom.symbol)
                    for atom in self.atoms]
        jsymbols = [elementlist.index(atom.symbol)
                    for atom in self.extendedatoms]
        ksymbols = [elementlist.index(atom.symbol)
                    for atom in self.extendedatoms]

        for data in self.av:
            i, j, k, fcij, fcjk, theta = data

            indexi = (index == i)
            indexj = (index == j % len(self.atoms))
            indexk = (index == k % len(self.atoms))

            if not (indexi or indexj or indexk):
                continue

            A = isymbols[i]
            B = jsymbols[j]
            C = ksymbols[k]

            diffvec = xvec - theta
            gaussian = np.exp(- diffvec**2 / 2 / self.ascale**2)

            # First term:
            first = np.zeros([self.nanglebins, 3])
            if indexi:
                first += (fcjk * np.outer(gaussian, self.nabla_fcij(i, j)))
            if indexj:
                first += (fcjk * np.outer(gaussian, -self.nabla_fcij(i, j)))

            # Second term:
            second = np.zeros([self.nanglebins, 3])
            if indexj:
                second += (fcij * np.outer(gaussian, self.nabla_fcjk(j, k)))
            if indexk:
                second += (fcij * np.outer(gaussian, -self.nabla_fcjk(j, k)))

            # Third term:
            third = np.zeros([self.nanglebins, 3])
            thirdinit = (fcij * fcjk * diffvec / self.ascale**2 * gaussian)

            if indexi:
                third += np.outer(thirdinit,
                                  self.dthetaijk_dri(i, j, k, theta))
            if indexj:
                third += np.outer(thirdinit,
                                  self.dthetaijk_drj(i, j, k, theta))
            if indexk:
                third += np.outer(thirdinit,
                                  self.dthetaijk_drk(i, j, k, theta))

            gradient[A, B, C] += ((first + second + third))

        if self.weight_by_elements:
            factortable = np.einsum('i,j,k->ijk',
                                    self.elcounts,
                                    self.elcounts,
                                    self.elcounts).astype(float)**-1
            gradient = np.einsum('ijklm,ijk->ijklm', gradient, factortable)

        return gradient * self.angleconstant

    def calculate_all_angle_gradients(self):
        self.anglegradients = [self.calculate_angle_gradient(atom.index)
                               for atom in self.atoms]
        return np.array(self.anglegradients)

    # ::: KERNEL STUFF ::: #
    # -------------------- #

    def distance(self, x1, x2):
        ''' Euclidean distance between two fingerprints '''
        return pdist([x1.get_fingerprint_vector(),
                      x2.get_fingerprint_vector()])

    def kernel(self, x1, x2):
        ''' Squared exponential kernel function '''
        return np.exp(-self.distance(x1, x2)**2 / 2 / self.l**2)

    def kernel_gradient(self, fp2, index):
        """
        Calculates the derivative of the kernel between
        self and fp2 with respect to atom with index 'index' in atom set
        of self.
        """

        result = self.dk_dD(fp2) * self.dD_drm(fp2, index)

        return result

    def dk_dD(self, fp2):
        ''' Derivative of the squared exponential kernel
        w.r.t. distance between fingerprints self and fp2 '''

        result = - (self.distance(self, fp2) / self.l**2
                    * self.kernel(self, fp2))

        return result

    def dD_drm(self, fp2, index):
        ''' Derivative of distance w.r.t. one of the coordinates in
        self. '''

        D = self.distance(self, fp2)
        if D == 0.0:
            return np.zeros(3)

        # Radial contribution:

        gs = self.gradients[index]
        A = list(self.elements).index(self.atoms[index].symbol)

        # difference vector between fingerprints:
        tildexvec = self.G - fp2.G

        Bsum = np.zeros(3)

        for B in range(self.n):
            Bsum += (1 + int(A != B)) * np.tensordot(tildexvec[B, A],
                                                     gs[B],
                                                     axes=[0, 0])

        result = Bsum

        # Angle contribution:

        gs = self.anglegradients[index]
        tildexvec = self.H - fp2.H
        summ = np.zeros(3)
        for A in range(self.n):
            for B in range(self.n):
                for C in range(self.n):
                    summ += np.tensordot(tildexvec[A, B, C],
                                         gs[A, B, C],
                                         axes=[0, 0])
        result += summ
        result /= D
        return result

    def kernel_hessian(self, fp2, index1, index2):

        D = self.distance(self, fp2)

        prefactor = 1 / self.l**2 * self.kernel(self, fp2)

        # Radial contribution:

        g1 = self.gradients[index1]
        g2 = fp2.gradients[index2]
        A1 = list(self.elements).index(self.atoms[index1].symbol)
        A2 = list(fp2.elements).index(fp2.atoms[index2].symbol)

        C1 = np.zeros([3, 3])
        for B in range(self.n):  # sum over elements
            if A1 == A2:
                C1 += ((1 + int(B != A2)) *
                       np.tensordot(g1[B], g2[B], axes=[0, 0]))
            else:
                if B in [A1, A2]:
                    C1 += np.tensordot(g1[A2], g2[A1], axes=[0, 0])

        # Angle contribution:

        g1 = self.anglegradients[index1]
        g2 = fp2.anglegradients[index2]
        C2 = np.zeros([3, 3])
        for A in range(self.n):
            for B in range(self.n):
                for C in range(self.n):
                    C2 += np.tensordot(g1[A, B, C], g2[A, B, C], axes=[0, 0])

        result = prefactor * (D**2 / self.l**2 *
                              np.outer(self.dD_drm(fp2, index1),
                                       fp2.dD_drm(self, index2)) +
                              C1 + C2)

        return result

    def Gij(self, i, j):
        ''' A function needed for derivatives of the radial fingerprint '''
        xij = self.dm[i, j]

        if xij == 0 or xij > self.limit:
            return 0.0

        xvec = np.linspace(0., self.limit, self.N)
        diffvec = xvec - xij
        Gij = ((2 / xij - diffvec / self.delta**2) *
               self.h[i, j] / xij *
               np.exp(- diffvec**2 / 2 / self.delta**2))

        return Gij


class OganovFP():

    def __init__(self, limit=10.0, Rlimit=4.0,
                 delta=0.5, N=200,
                 pbc=None, calc_gradients=True,
                 weight=1.0, scale=1.0):

        self.limit = limit
        self.delta = delta
        self.N = N
        self.pbc = pbc

        self.gamma = 2

        # modifiable parameters:
        self.params = {'weight': weight,
                       'scale': scale,
                       'delta': self.delta}

        self.calc_gradients = calc_gradients
        self.weight_by_elements = True
        self.set_params()

    def set_params(self):
        self.weight = self.params['weight']
        self.l = self.params['scale']
        self.delta = self.params['delta']

        return

    def set_atoms(self, atoms):
        ''' Set new atoms and initialize '''

        self.atoms = atoms
        self.Vuc = self.atoms.get_volume()

        if self.pbc is None:
            self.pbc = self.atoms.pbc
        elif self.pbc is False:
            self.pbc = np.array([False, False, False])

        self.elements = np.sort(list(set([atom.symbol for atom in atoms])))
        self.elcounts = [len([atom for atom in self.atoms if
                             atom.symbol == self.elements[i]])
                         for i in range(len(self.elements))]

        self.n = len(self.elements)

        self.extend_positions()
        self.update()

    def update(self, params=None):
        ''' Update method when parameters are changed '''

        if params is not None:
            for param in params:

                self.params[param] = params[param]

        self.set_params()

        self.set_peak_heights()
        self.get_fingerprint()

        if self.calc_gradients:
            self.calculate_all_gradients()

        self.dFP_dDelta_calculated = False
        self.dGij_dDelta_calculated = False
        self.d_dDelta_dFP_drm_calculated = False

    def extend_positions(self):
        ''' Extend the unit cell so that all the atoms within the limit
        are in the same cell, indexed properly.
        '''

        # Determine unit cell parameters:
        cell = self.atoms.cell.array
        lengths = self.atoms.cell.lengths()
        natoms = len(self.atoms)

        self.origcell = cell

        # Number of cells needed to consider given the limit and pbc:
        ncells = [self.limit // lengths[i] + 1 for i in range(3)]
        nx, ny, nz = [1 + 2 * int(n) * self.pbc[i]
                      for i, n in enumerate(ncells)]

        self.extendedatoms = self.atoms.repeat([nx, ny, nz])

        newstart = natoms * int(np.prod([nx, ny, nz]) / 2)
        newend = newstart + natoms
        self.atoms = self.extendedatoms[newstart:newend]

        # Distance matrix
        self.dm = distance_matrix(x=self.atoms.positions,
                                  y=self.extendedatoms.positions)

        # position vector matrix
        self.rm = np.einsum('ilkl->ikl',
                            np.subtract.outer(self.atoms.positions,
                                              self.extendedatoms.positions))

        return

    def set_peak_heights(self):
        ''' Calculate the delta peak heights self.h '''

        self.constant = 1 / (self.limit / self.N)

        # Ignore zero-division warning
        with catch_warnings():
            simplefilter("ignore", category=RuntimeWarning)
            self.h = np.where(self.dm > 0.0001,
                              (self.constant / self.dm**2),
                              0.0)

        return self.h

    def get_fingerprint(self):
        ''' Calculate the Gaussian-broadened fingerprint. '''

        self.G = np.ndarray([self.n, self.n, self.N])
        x = np.linspace(0, self.limit, self.N)  # variable array

        # Broadening of each peak:
        for i in range(self.n):
            for j in range(self.n):

                # Get peak positions
                R = np.where(self.dm < self.limit,
                             self.dm,
                             0.0)

                # Get peak heights
                h = np.where(self.dm < self.limit,
                             self.h,
                             0.0)

                # Consider only the correct elements i and j
                ms = (np.array(self.atoms.get_chemical_symbols()) ==
                      self.elements[i])
                ns = (np.array(self.extendedatoms.get_chemical_symbols()) ==
                      self.elements[j])

                R = (R.T * ms).T
                R = R * ns
                R = R.flatten()

                h = (h.T * ms).T
                h = h * ns
                h = h.flatten()

                # Remove zero-height peaks
                nz = np.nonzero(h)
                R = R[nz]
                h = h[nz]

                npeaks = len(R)  # number of peaks

                g = np.zeros(self.N)
                for p in range(npeaks):
                    g += h[p] * np.exp(- (x - R[p])**2 / 2 / self.delta**2)

                self.G[i, j] = g

        if self.weight_by_elements:
            factortable = np.einsum('i,j->ij',
                                    self.elcounts,
                                    self.elcounts).astype(float)**-1
            self.G = np.einsum('ijk,ij->ijk', self.G, factortable)

        return self.G

    def get_fingerprint_vector(self):
        return self.G.flatten()

    # ::: GRADIENTS ::: #
    # ----------------- #

    def calculate_gradient(self, index):
        '''
        Calculates the derivative of the fingerprint
        with respect to one of the coordinates.

        index: Atom index with which to differentiate
        '''
        gradient = np.zeros([self.n, self.N, 3])

        i = index
        A = list(self.elements).index(self.atoms[i].symbol)

        elementlist = list(self.elements)
        jsymbols = [elementlist.index(atom.symbol)
                    for atom in self.extendedatoms]
        # Sum over elements:
        for B in range(self.n):

            jsum = np.zeros([self.N, 3])

            # sum over atoms in extended cell
            for j in range(len(self.extendedatoms)):

                # atom j is of element B:
                if B != jsymbols[j]:
                    continue

                # position vector between atoms:
                rij = self.rm[i, j]
                Gij = self.Gij(i, j)
                jsum += np.outer(Gij, -rij)

            gradient[B] = (1 + int(A == B)) * jsum

        if self.weight_by_elements:
            factortable = (self.elcounts[A] * 
                           np.array(self.elcounts)).astype(float)**-1
            gradient = [gradient[i] * factortable[i]
                        for i in range(len(gradient))]

        return gradient

    def calculate_all_gradients(self):
        self.gradients = np.array([self.calculate_gradient(atom.index)
                                   for atom in self.atoms])
        return self.gradients

    # ::: KERNEL STUFF ::: #
    # -------------------- #

    def distance(self, x1, x2):
        return pdist([x1.get_fingerprint_vector(),
                      x2.get_fingerprint_vector()])

    def kernel(self, x1, x2):
        return np.exp(-self.distance(x1, x2)**2 / 2 / self.l**2)

    def kernel_gradient(self, fp2, index):
        """
        Calculates the derivative of the kernel between
        self and fp2 with respect to atom with index 'index' in atom set
        of self.
        """

        result = self.dk_dD(fp2) * self.dD_drm(fp2, index)

        return result

    def dk_dD(self, fp2):
        result = - (self.distance(self, fp2) / self.l**2
                    * self.kernel(self, fp2))

        return result

    def dD_drm(self, fp2, index):

        D = self.distance(self, fp2)
        if D == 0.0:
            return np.zeros(3)

        gs = self.gradients[index]
        A = list(self.elements).index(self.atoms[index].symbol)

        # difference vector between fingerprints:
        tildexvec = self.G - fp2.G

        Bsum = np.zeros(3)

        for B in range(self.n):
            Bsum += (1 + int(A != B)) * np.tensordot(tildexvec[B, A],
                                                     gs[B],
                                                     axes=[0, 0])

        result = Bsum / D
        return result

    def kernel_hessian(self, fp2, index1, index2):

        D = self.distance(self, fp2)

        prefactor = 1 / self.l**2 * self.kernel(self, fp2)

        g1 = self.gradients[index1]
        g2 = fp2.gradients[index2]
        A1 = list(self.elements).index(self.atoms[index1].symbol)
        A2 = list(fp2.elements).index(fp2.atoms[index2].symbol)

        C1 = np.zeros([3, 3])
        for B in range(self.n):  # sum over elements
            if A1 == A2:
                C1 += ((1 + int(B != A2)) *
                       np.tensordot(g1[B], g2[B], axes=[0, 0]))
            else:
                if B in [A1, A2]:
                    C1 += np.tensordot(g1[A2], g2[A1], axes=[0, 0])

        result = prefactor * (D**2 / self.l**2 *
                              np.outer(self.dD_drm(fp2, index1),
                                       fp2.dD_drm(self, index2)) +
                              C1)

        return result

    # ---------------------------------------------------------
    # ------------- Derivatives w.r.t. Delta ------------------
    # ---------------------------------------------------------

    def dk_dDelta(self, fp2):
        return self.dk_dD(fp2) * self.dD_dDelta(fp2)

    def dD_dDelta(self, fp2):
        D = self.distance(self, fp2)
        if D == 0:
            return 0

        result = 0
        dFP_dDelta1 = self.dFP_dDelta()
        dFP_dDelta2 = fp2.dFP_dDelta()
        for A in range(self.n):
            for B in range(self.n):
                first = self.G[A, B] - fp2.G[A, B]
                second = dFP_dDelta1[A, B] - dFP_dDelta2[A, B]
                result += first.dot(second)

        result *= 1 / D

        return result

    def dFP_dDelta(self):

        if self.dFP_dDelta_calculated:
            return self.dfp_ddelta

        xvec = np.linspace(0., self.limit, self.N)
        result = np.zeros([self.n, self.n, self.N])

        elementlist = list(self.elements)
        isymbols = [elementlist.index(atom.symbol)
                    for atom in self.atoms]
        jsymbols = [elementlist.index(atom.symbol)
                    for atom in self.extendedatoms]

        for A in range(self.n):
            for B in range(A, self.n):

                for i in range(len(self.atoms)):
                    if A != isymbols[i]:
                        continue

                    for j in range(len(self.extendedatoms)):
                        if B != jsymbols[j]:
                            continue

                        xij = self.dm[i, j]

                        if xij == 0 or xij > self.limit:
                            continue

                        normsq = (xvec - xij)**2
                        subresult = np.exp(- normsq / 2 / self.delta**2)
                        subresult *= ((normsq / self.delta**2 - 1) *
                                      self.h[i, j])

                        result[A, B] += subresult
                result[B, A] = result[A, B]  # symmetric matrix

        result *= 1 / self.delta

        self.dfp_ddelta = result
        self.dFP_dDelta_calculated = True
        return result

    # ----------------------------------------------
    # d_dDelta: Gradient:

    def dk_drm_dDelta(self, fp2, index):

        D = self.distance(self, fp2)
        first = -D / self.l**2 * self.dk_dDelta(fp2) * self.dD_drm(fp2, index)

        prefactor = -1 / self.l**2 * self.kernel(self, fp2)

        i = index
        A = list(self.elements).index(self.atoms[index].symbol)

        dFP_dDelta1 = self.dFP_dDelta()
        dFP_dDelta2 = fp2.dFP_dDelta()

        Bsum = np.zeros(3)
        for B in range(self.n):
            tildexvec = dFP_dDelta1[A, B] - dFP_dDelta2[A, B]
            jsum = self.gradients[index][B]
            Bsum += ((1 + int(A != B)) *
                     np.tensordot(tildexvec, jsum, axes=[0, 0]))

        second = prefactor * Bsum

        elementlist = list(self.elements)
        jsymbols = [elementlist.index(atom.symbol)
                    for atom in self.extendedatoms]
        Bsum = np.zeros(3)
        # Sum over elements:
        for B in range(self.n):
            tildexvec = self.G[A, B] - fp2.G[A, B]
            jsum = np.zeros([self.N, 3])

            for j in range(len(self.extendedatoms)):

                if B != jsymbols[j]:
                    continue

                rij = self.rm[i, j]
                jsum += ((1 + int(A == B)) *
                         np.outer(self.dGij_dDelta(i, j), -rij))

            Bsum += ((1 + int(A != B)) *
                     np.tensordot(tildexvec, jsum, axes=[0, 0]))

        third = prefactor * Bsum

        return first + second + third

    def Gij(self, i, j):
        xij = self.dm[i, j]

        if xij == 0 or xij > self.limit:
            return 0.0

        xvec = np.linspace(0., self.limit, self.N)
        diffvec = xvec - xij
        Gij = ((2 / xij - diffvec / self.delta**2) *
               self.h[i, j] / xij *
               np.exp(- diffvec**2 / 2 / self.delta**2))

        return Gij

    def dGij_dDelta(self, i, j):

        xvec = np.linspace(0., self.limit, self.N)
        xij = self.dm[i, j]

        if xij == 0 or xij > self.limit:
            return 0

        normsq = (xvec - xij)**2

        first = (1 / self.delta * self.h[i, j] / xij *
                 np.exp(-normsq / 2 / self.delta**2))
        second = (2 / xij * (normsq / self.delta**2 - 1) +
                  (xvec - xij) / self.delta**2 *
                  (3 - normsq / self.delta**2))

        return first * second

    # ----------------------------------------------
    # d_dDelta: Hessian

    def dk_drm_drn_dDelta(self, fp2, index1, index2):

        dD_dDelta = self.dD_dDelta(fp2)
        D = self.distance(self, fp2)
        first = - D / self.l**2 * dD_dDelta * self.kernel_hessian(fp2,
                                                                  index1,
                                                                  index2)

        prefactor = 1 / self.l**2 * self.kernel(self, fp2)

        second = D * np.outer(self.d_dDelta_D_dD_drm(fp2, index1),
                              fp2.dD_drm(self, index2))
        second += D * np.outer(self.dD_drm(fp2, index1),
                               fp2.d_dDelta_D_dD_drm(self, index2))
        second *= prefactor / self.l**2

        third = np.zeros([3, 3])
        A1 = list(self.elements).index(self.atoms[index1].symbol)
        A2 = list(fp2.elements).index(fp2.atoms[index2].symbol)
        d1 = self.d_dDelta_dFP_drm(index1)
        d2 = fp2.d_dDelta_dFP_drm(index2)

        for B in range(self.n):
            if A1 == A2:
                prefactor2 = (1 + int(A1 != B))
                third += prefactor2 * np.tensordot(d1[B],
                                                   fp2.gradients[index2][B],
                                                   axes=[0, 0])
                third += prefactor2 * np.tensordot(self.gradients[index1][B],
                                                   d2[B],
                                                   axes=[0, 0])
            else:
                if B in [A1, A2]:
                    third += np.tensordot(d1[A2],
                                          fp2.gradients[index2][A1],
                                          axes=[0, 0])
                    third += np.tensordot(self.gradients[index1][A2],
                                          d2[A1],
                                          axes=[0, 0])
        third *= prefactor

        return first + second + third

    def d_dDelta_D_dD_drm(self, fp2, index):
        i = index
        A = list(self.elements).index(self.atoms[i].symbol)

        Bsum = np.zeros(3)
        # Sum over elements:
        dFP_dDelta1 = self.dFP_dDelta()
        dFP_dDelta2 = fp2.dFP_dDelta()

        d2 = self.d_dDelta_dFP_drm(i)

        g = self.gradients[i]

        for B in range(self.n):
            tildexvec = self.G[A, B] - fp2.G[A, B]
            tildexvec_dDelta = dFP_dDelta1[A, B] - dFP_dDelta2[A, B]
            prefactor = 1 + int(A != B)
            Bsum += prefactor * np.tensordot(tildexvec_dDelta,
                                             g[B],
                                             axes=[0, 0])
            Bsum += prefactor * np.tensordot(tildexvec,
                                             d2[B],
                                             axes=[0, 0])

        return Bsum

    def d_dDelta_dFP_drm(self, index):

        if self.d_dDelta_dFP_drm_calculated:
            return self.d_ddelta_dfp_drm[index]

        self.d_ddelta_dfp_drm = np.zeros([len(self.atoms),
                                          self.n,
                                          self.N,
                                          3])
        elementlist = list(self.elements)
        isymbols = np.array([elementlist.index(atom.symbol)
                             for atom in self.atoms])
        jsymbols = np.array([elementlist.index(atom.symbol)
                             for atom in self.extendedatoms])
        factors = np.ones([self.n, self.n]) + np.eye(self.n)

        for i in range(len(self.atoms)):

            A = isymbols[i]

            for B in range(self.n):

                jsum = np.zeros([self.N, 3])

                for j in range(len(self.extendedatoms)):

                    if B != jsymbols[j]:
                        continue

                    jsum += np.outer(self.dGij_dDelta(i, j), -self.rm[i, j])

                self.d_ddelta_dfp_drm[i][B] += factors[A, B] * jsum

        self.d_dDelta_dFP_drm_calculated = True
        return self.d_ddelta_dfp_drm[index]

    # Derivatives w.r.t. scale (l):
    def dk_dl(self, fp2):
        result = (self.distance(self, fp2)**2
                  * 1 / self.l**3
                  * self.kernel(self, fp2))
        return result

    def d_dl_dk_drm(self, fp2, index):
        result = (1 / self.l
                  * (self.distance(self, fp2)**2 / self.l**2 - 2)
                  * self.dk_dD(fp2)
                  * self.dD_drm(fp2, index))
        return result

    def d_dl_dk_drm_drn(self, fp2, index1, index2):
        Dscaled_squared = (self.distance(self, fp2) / self.l)**2
        first = (1 / self.l
                 * (Dscaled_squared - 2)
                 * self.kernel_hessian(fp2, index1, index2))

        second = -(2 / self.l**3
                   * self.kernel(self, fp2)
                   * Dscaled_squared
                   * np.outer(self.dD_drm(fp2, index1),
                              fp2.dD_drm(self, index2)))

        return first + second

    def dk_dweight(self, fp2):
        return self.kernel(self, fp2) * 2


class CartesianCoordFP():

    def __init__(self):
        return

    def set_atoms(self, atoms):
        self.atoms = atoms

    def get_fingerprint_vector(self):
        return self.atoms.get_positions(wrap=False).reshape(-1)
