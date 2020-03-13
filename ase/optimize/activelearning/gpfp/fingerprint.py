from scipy.spatial.distance import pdist
from math import pi
from scipy.spatial import distance_matrix
import numpy as np
from numpy import dot
from warnings import catch_warnings, simplefilter
from ase.parallel import world

class Fingerprint():
    ''' Master class for structural fingerprints.
    A fingerprint class should deprecate each of the methods
    defined here.'''

    def __init__(self, calc_gradients=True):
        return

    def set_params(self, params):
        '''
        params: dict
        '''
        return

    def set_atoms(self, atoms):
        '''
        atoms: ase.Atoms object
        '''
        return

    def update(self, params):
        '''
        params: dict
        '''
        return

    def kernel(self, fp2):
        '''
        fp2: Fingerprint object
        '''
        return

    def kernel_gradient(self, fp2, index):
        '''
        fp2: Fingerprint object
        index: int
        '''
        return

    def kernel_hessian(self, fp2, index1, index2):
        '''
        fp2: Fingerprint object
        index1: int
        index2: int
        '''
        return


class OganovFP(Fingerprint):

    def __init__(self, pbc=None, calc_gradients=True,
                 weight_by_elements=True, **kwargs):
        ''' Parameters:

        limit: float
               Threshold for radial fingerprint (Angstroms)

        delta: float
               Width of Gaussian broadening in radial fingerprint
               (Angstroms)

        N: int
           Number of bins in radial fingerprint

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
        TODO: Get rid of weight and scale

        '''

        default_parameters = {'weight': 1.0,
                              'scale': 1.0,
                              'limit': 20.0,
                              'delta': 0.2,
                              'N': 200}

        self.params = default_parameters.copy()
        self.params.update(kwargs)

        self.pbc = pbc
        self.calc_gradients = calc_gradients
        self.weight_by_elements = weight_by_elements

        self.set_params()

    def set_params(self):
        self.l = self.params.get('scale')
        self.delta = self.params.get('delta')
        self.limit = self.params.get('limit')
        self.N = self.params.get('N')

        return

    def set_atoms(self, atoms):
        ''' Set new atoms and initialize '''

        self.atoms = atoms
        self.initialize()
        self.extend_positions()
        self.update()

    def initialize(self):
        ''' Initialize pbc and elements '''

        if self.pbc is None:
            self.pbc = self.atoms.pbc
        elif self.pbc is False:
            self.pbc = np.array([False, False, False])
        elif self.pbc is True:
            self.pbc = np.array([True, True, True])

        self.elements = np.sort(list(set([atom.symbol
                                          for atom in self.atoms])))
        self.elcounts = [len([atom for atom in self.atoms if
                              atom.symbol == self.elements[i]])
                         for i in range(len(self.elements))]

        self.n = len(self.elements)

    def update(self, params=None):
        ''' Update method when parameters are changed '''

        if params is not None:
            self.params.update(params)

        self.set_params()

        self.set_peak_heights()
        self.get_fingerprint()

        if self.calc_gradients:
            self.calculate_all_gradients()

        self.dFP_dDelta_calculated = False
        self.dGij_dDelta_calculated = False
        self.d_dDelta_dFP_drm_calculated = False

        self.vector = self.G.flatten()

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
        ap = self.atoms.positions
        ep = self.extendedatoms.positions
        dm = distance_matrix(x=ap, y=ep)

        mask = np.logical_or(dm == 0, dm > self.limit)
        r_indices = []
        for i in range(len(self.atoms)):
            for j in range(len(self.extendedatoms)):

                if mask[i, j]:
                    continue

                r_indices.append((i, j))
        self.r_indices = np.array(r_indices, dtype=int)

        # position vector matrix
        self.rm = ap[self.r_indices[:, 0]] - ep[self.r_indices[:, 1]]
        self.dm = np.linalg.norm(self.rm, axis=1)

        elementlist = list(self.elements)
        self.prim_symbols = [elementlist.index(atom.symbol)
                             for atom in self.atoms]
        self.ext_symbols = [elementlist.index(atom.symbol)
                            for atom in self.extendedatoms]
        self.AB = np.array([(self.prim_symbols[i], self.ext_symbols[j])
                            for i, j in self.r_indices], dtype=int)
        return

    def set_peak_heights(self):
        ''' Calculate the delta peak heights self.h '''

        self.constant = 1 / (self.limit / self.N)
        self.h = self.constant * (1 / self.dm**2 -
                                  1 / self.limit**2)
        return

    def get_fingerprint(self):
        ''' Calculate the Gaussian-broadened fingerprint. '''

        self.G = np.zeros([self.n, self.n, self.N])
        x = np.linspace(0, self.limit, self.N)  # variable array

        # Broadening of each peak:
        for p in range(len(self.r_indices)):
            i, j = self.r_indices[p]

            h = self.h[p]
            R = self.dm[p]
            g = h * np.exp(- (x - R)**2 / 2 / self.delta**2)
            A, B = self.AB[p]
            self.G[A, B] += g
                
        if self.weight_by_elements:
            factortable = np.einsum('i,j->ij',
                                    self.elcounts,
                                    self.elcounts).astype(float)**-1
            self.G = np.einsum('ijk,ij->ijk', self.G, factortable)

        return self.G

    def get_fingerprint_vector(self):
        return self.vector

    # ::: GRADIENTS ::: #
    # ----------------- #

    def calculate_gradient(self, index):
        '''
        Calculates the derivative of the fingerprint
        with respect to one of the coordinates.

        index: Atom index with which to differentiate
        '''
        gradient = np.zeros([self.n, self.n, self.N, 3])
        n = len(self.atoms)

        mask = np.arange(n) == index
        ext_mask = np.arange(len(self.extendedatoms)) % n == index

        for p in range(len(self.r_indices)):
            i, j = self.r_indices[p]
            indexi = mask[i]
            indexj = ext_mask[j]

            if not (indexi or indexj):
                continue

            if indexi and indexj:
                continue
            
            # position vector between atoms:
            rij = self.rm[p]
            Gij = self.Gij(p)

            if indexj:
                rij = -rij

            A, B = self.AB[p]
            gradient[A, B] += np.outer(Gij, -rij)

        if self.weight_by_elements:
            factortable = np.einsum('i,j->ij',
                                    self.elcounts,
                                    self.elcounts).astype(float)**-1
            gradient = np.einsum('ijkl,ij->ijkl', gradient, factortable)

        return gradient

    def calculate_all_gradients(self):

        self.gradients = np.array([self.calculate_gradient(atom.index)
                                   for atom in self.atoms])
        return self.gradients

    # ::: KERNEL STUFF ::: #
    # -------------------- #

    def distance(self, x1, x2):
        ''' Distance function between two fingerprints '''
        return pdist([x1.get_fingerprint_vector(),
                      x2.get_fingerprint_vector()])

    def kernel(self, x1, x2):
        ''' Squared Exponential kernel function using some
        distance function '''
        return np.exp(-self.distance(x1, x2)**2 / 2 / self.l**2)

    def kernel_gradient(self, fp2, index, kernel=None, D=None, dD_dr=None):
        """
        Calculates the derivative of the kernel between
        self and fp2 with respect to atom with index 'index' in atom set
        of self using chain rule.
                        d k(x, x')    dk      d D(x, x')
                       ----------- = ----  X  ----------
                           d xi       dD         d xi
        """

        if dD_dr is None:
            dD_dr = self.dD_drm(fp2, index, D)

        result = self.dk_dD(fp2, kernel, D) * dD_dr

        return result

    def dk_dD(self, fp2, kernel=None, D=None):
        ''' Derivative of kernel function w.r.t. distance function
            dk / dD
        '''

        if kernel is None:
            kernel = self.kernel(self, fp2)

        if D is None:
            D = self.distance(self, fp2)

        result = - (D / self.l**2 * kernel)

        return result

    def dD_drm(self, fp2, index, D=None):
        ''' Gradient of distance function:

                      d D(x, x')
                      ----------
                         d xi
        '''

        if D is None:
            D = self.distance(self, fp2)

        if D == 0.0:
            return np.zeros(3)

        g = self.gradients[index]

        # difference vector between fingerprints:
        tildexvec = self.G - fp2.G

        return 1 / D * np.einsum('ijk,ijkl->l', tildexvec, g)

    def kernel_hessian(self, fp2, index1, index2, kernel=None, D=None, dD_dr1=None, dD_dr2=None):
        ''' Squared exponential kernel hessian w.r.t. atomic
        coordinates, ie.
                            d^2 k(x, x')
                           -------------
                             dx_i dx_j
        '''
        if D is None:
            D = self.distance(self, fp2)

        if kernel is None:
            kernel = self.kernel(self, fp2)

        if dD_dr1 is None:
            dD_dr1 = self.dD_drm(fp2, index1, D=D)

        if dD_dr2 is None:
            dD_dr2 = fp2.dD_drm(self, index2, D=D)

        prefactor = 1 / self.l**2 * kernel

        g1 = self.gradients[index1]
        g2 = fp2.gradients[index2]
        C1 = np.einsum('ijkl,ijkm->lm', g1, g2)

        result = prefactor * (D**2 / self.l**2 *
                              np.outer(dD_dr1, dD_dr2) + C1)

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

    def Gij(self, p):
        xij = self.dm[p]
        xvec = np.linspace(0., self.limit, self.N)
        diffvec = xvec - xij

        h = self.h[p]
        h_bare = self.h[p] + self.constant / self.limit**2
        Gij = ((2 * h_bare / xij - diffvec * h / self.delta**2) / xij *
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


class RadialAngularFP(OganovFP):

    def __init__(self, pbc=None, calc_gradients=True,
                 weight_by_elements=True, **kwargs):
        ''' Parameters:

        Rlimit: float
                Threshold for angular fingerprint (Angstroms)

        ascale: float
                Width of Gaussian broadening in angular fingerprint
               (Radians)

        Na: int
            Number of bins in angular fingerprint

        aweight: float
            Scaling factor for the angular fingerprint; the angular
            fingerprint is multiplied by this number

        TODO: Rename class attributes.

        '''

        default_parameters = {'Rlimit': 4.0,
                              'ascale': 0.2,
                              'Na': 100,
                              'aweight': 1.0}

        self.params = default_parameters.copy()
        self.params.update(kwargs)
        OganovFP.__init__(self, pbc=pbc, calc_gradients=calc_gradients,
                          weight_by_elements=weight_by_elements,
                          **self.params)
        
        assert self.limit >= self.Rtheta

        self.gamma = 2

    def set_params(self):
        ''' Set parameters according to dictionary
            self.params '''

        self.weight = self.params.get('weight')
        self.l = self.params.get('scale')
        self.limit = self.params.get('limit')
        self.Rtheta = self.params.get('Rlimit')
        self.delta = self.params.get('delta')
        self.ascale = self.params.get('ascale')
        self.aweight = self.params.get('aweight')
        self.N = self.params.get('N')
        self.nanglebins = self.params.get('Na')

        return

    def set_atoms(self, atoms):
        ''' Set new atoms and initialize '''

        self.atoms = atoms
        self.initialize()
        self.extend_positions()
        self.set_angles()
        self.update()

    def update(self, params=None):
        ''' Update method when parameters are changed '''

        if params is not None:
            self.params.update(params)

        self.set_params()
        self.set_peak_heights()
        self.get_fingerprint()

        if self.calc_gradients:
            self.calculate_all_gradients()

        self.get_angle_fingerprint()

        if self.calc_gradients:
            self.calculate_all_angle_gradients()
        
        self.vector = np.concatenate((self.G.flatten(), self.H.flatten()), axis=None)

    def set_angles(self):
        """
        In angle vector 'self.av' all angles are saved where
        one of the atoms is in 'self.atoms' and the other
        two are in 'self.extendedatoms'
        """

        # Extended distance and displacement vector matrices:
        ap = self.atoms.positions
        ep = self.extendedatoms.positions
        edm = distance_matrix(ep, ep)

        self.angleconstant = self.aweight / (pi / self.nanglebins)

        indices = []

        dm = distance_matrix(ap, ep)
        
        mask1 = np.logical_or(dm == 0, dm > self.Rtheta)
        mask2 = dm == 0
        mask3 = np.logical_or(edm == 0, edm > self.Rtheta)

        for i in range(len(self.atoms)):
            for j in range(len(self.extendedatoms)):

                if mask1[i, j]:
                    continue

                for k in range(len(self.extendedatoms)):

                    if mask2[i, k]:
                        continue

                    if mask3[j, k]:
                        continue

                    indices.append((i, j, k))

        self.indices = np.array(indices, dtype=int)

        if len(self.indices) == 0:
            self.arm = np.array([[]])
            self.erm = np.array([[]])
            self.adm = np.array([])
            self.edm = np.array([])
            args = np.array([])
            self.fcij = np.array([])
            self.fcjk = np.array([])
            self.thetas = np.array([])
            self.ABC = np.array([[]])

        else:
            self.arm = ap[self.indices[:, 0]] - ep[self.indices[:, 1]]
            self.erm = ep[self.indices[:, 2]] - ep[self.indices[:, 1]]
            self.adm = np.linalg.norm(self.arm, axis=1)
            self.edm = np.linalg.norm(self.erm, axis=1)

            args = np.einsum('ij,ij->i', self.arm, self.erm) / self.adm / self.edm

            # Take care of numerical errors:
            args = np.where(args >= 1.0, 1.0 - 1e-9, args)
            args = np.where(args <= -1.0, -1.0 + 1e-9, args)

            self.fcij = self.cutoff_function(self.adm)
            self.fcjk = self.cutoff_function(self.edm)
            self.thetas = np.arccos(args)

            self.ABC = np.array([(self.prim_symbols[i],
                                  self.ext_symbols[j],
                                  self.ext_symbols[k])
                                 for i, j, k in self.indices], dtype=int)
        
        return

    def cutoff_function(self, r):
        """
        Rtheta: cutoff radius, given in angstroms
        """

        return np.where(r <= self.Rtheta,
                        (1 + self.gamma * (r / self.Rtheta)**(self.gamma + 1) -
                         (self.gamma + 1) * (r / self.Rtheta)**self.gamma),
                        0.0)

    def get_angle_fingerprint(self):
        ''' Calculate the angular fingerprint with Gaussian broadening  '''

        self.H = np.zeros([self.n, self.n, self.n, self.nanglebins])
        x = np.linspace(0, pi, self.nanglebins)  # variable array

        # Broadening of each peak:
        for p in range(len(self.indices)):
            i, j, k = self.indices[p]
            fcij = self.fcij[p]
            fcjk = self.fcjk[p]
            theta = self.thetas[p]

            A, B, C = self.ABC[p]
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
        ''' Return the full fingerprint vector with Oganov part and
        angular distribution. '''

        return self.vector

    # ::: GRADIENTS ::: #
    # ----------------- #

    def nabla_fcij(self):
        d = self.adm
        r = self.arm
        dfc_dd = (self.gamma * (self.gamma + 1) / self.Rtheta *
                  ((d / self.Rtheta) ** self.gamma -
                   (d / self.Rtheta) ** (self.gamma - 1)))
        dd_drm = np.einsum('ij,i->ij', r, d**-1)
        return np.einsum('i,ij->ij', dfc_dd, dd_drm) 

    def nabla_fcjk(self):
        d = self.edm
        r = -self.erm # in parallel version, erm[m,n] are not calcd
        dfc_dd = (self.gamma * (self.gamma + 1) / self.Rtheta *
                  ((d / self.Rtheta) ** self.gamma -
                   (d / self.Rtheta) ** (self.gamma - 1)))
        dd_drm = np.einsum('ij,i->ij', r, d**-1)
        return np.einsum('i,ij->ij', dfc_dd, dd_drm) 

    def dthetaijk_dri(self):
        r1 = self.adm
        v1 = self.arm
        r2 = self.edm
        v2 = self.erm
        dotp = np.einsum('ij,ij->i', v1, v2)
        prefs = 1 / abs(np.sin(self.thetas))
        return (prefs[:, np.newaxis] /
                r1[:, np.newaxis] /
                r2[:, np.newaxis] *
                (dotp[:, np.newaxis] / np.square(r1)[:, np.newaxis] * v1 - v2))


    def dthetaijk_drj(self):
        r1 = self.adm
        v1 = self.arm
        r2 = self.edm
        v2 = self.erm
        dotp = np.einsum('ij,ij->i', v1, v2)
        prefs = -1 / abs(np.sin(self.thetas))
        first = (-1 + dotp[:, np.newaxis] / np.square(r1)[:, np.newaxis]) * v1
        second = (-1 + dotp[:, np.newaxis] / np.square(r2)[:, np.newaxis]) * v2

        return (prefs[:, np.newaxis] /
                r1[:, np.newaxis] /
                r2[:, np.newaxis] *
                (first + second))

    def dthetaijk_drk(self):
        r1 = self.adm
        v1 = self.arm
        r2 = self.edm
        v2 = self.erm
        dotp = np.einsum('ij,ij->i', v1, v2)
        prefs = -1 / abs(np.sin(self.thetas))
        return (prefs[:, np.newaxis] /
                r1[:, np.newaxis] /
                r2[:, np.newaxis] *
                (v1 - dotp[:, np.newaxis] / np.square(r2)[:, np.newaxis] * v2))

    def calculate_angle_gradient(self, index,
                                 firstvalues, secondvalues,
                                 third_i, third_j, third_k):
        '''
        Calculates the derivative of the fingerprint
        with respect to one of the coordinates.

        index: Atom index with which to differentiate
        '''
        gradient = np.zeros([self.n, self.n, self.n, self.nanglebins, 3])

        n = len(self.atoms)
        mask = np.arange(n) == index
        ext_mask = np.arange(len(self.extendedatoms)) % n == index

        for p in range(len(self.indices)):
            i, j, k = self.indices[p]
            indexi = mask[i]
            indexj = ext_mask[j]
            indexk = ext_mask[k]

            if not (indexi or indexj or indexk):
                continue

            result = np.zeros([self.nanglebins, 3])
            if indexi:
                result += firstvalues[p] + third_i[p]
            if indexj:
                result += -firstvalues[p] + secondvalues[p] + third_j[p]
            if indexk:
                result += -secondvalues[p] + third_k[p]

            A, B, C = self.ABC[p]
            gradient[A, B, C] += result

        if self.weight_by_elements:
            factortable = np.einsum('i,j,k->ijk',
                                    self.elcounts,
                                    self.elcounts,
                                    self.elcounts).astype(float)**-1
            gradient = np.einsum('ijklm,ijk->ijklm', gradient, factortable)

        return gradient * self.angleconstant

    def calculate_all_angle_gradients(self):
        xvec = np.linspace(0., pi, self.nanglebins)
        diffvecs = np.subtract.outer(xvec, self.thetas).T
        gaussians = np.exp(- diffvecs**2 / 2 / self.ascale**2)
        nabla_fcijs = self.nabla_fcij()
        nabla_fcjks = self.nabla_fcjk()

        firstvalues = (self.fcjk[:, np.newaxis, np.newaxis] *
                       np.einsum('ij,ik->ijk', gaussians, nabla_fcijs))

        secondvalues = (self.fcij[:, np.newaxis, np.newaxis] *
                        np.einsum('ij,ik->ijk', gaussians, nabla_fcjks))

        thirdinits = (self.fcij[:, np.newaxis] *
                      self.fcjk[:, np.newaxis] *
                      diffvecs / self.ascale**2 * gaussians)
        dt_dris = self.dthetaijk_dri()
        dt_drjs = self.dthetaijk_drj()
        dt_drks = self.dthetaijk_drk()
        third_i = np.einsum('ij,ik->ijk', thirdinits, dt_dris)
        third_j = np.einsum('ij,ik->ijk', thirdinits, dt_drjs)
        third_k = np.einsum('ij,ik->ijk', thirdinits, dt_drks)

        del xvec, diffvecs, gaussians, nabla_fcijs, nabla_fcjks, dt_dris, dt_drjs, dt_drks

        self.anglegradients = [self.calculate_angle_gradient(atom.index,
                                                             firstvalues,
                                                             secondvalues,
                                                             third_i,
                                                             third_j,
                                                             third_k)
                               for atom in self.atoms]
        return np.array(self.anglegradients)

    # ::: KERNEL STUFF ::: #
    # -------------------- #

    def dD_drm(self, fp2, index, D=None):
        ''' Gradient of distance function:

                      d D(x, x')
                      ----------
                         d xi
        '''
        if D is None:
            D = self.distance(self, fp2)

        if D == 0.0:
            return np.zeros(3)

        # Radial contribution:
        result = OganovFP.dD_drm(self, fp2, index, D=D)

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
        result += summ / D
        return result

    def kernel_hessian(self, fp2, index1, index2, kernel=None, D=None, dD_dr1=None, dD_dr2=None):
        ''' Squared exponential kernel hessian w.r.t. atomic
        coordinates, ie.
                            d^2 k(x, x')
                           -------------
                             dx_i dx_j
        '''

        if kernel is None:
            kernel = self.kernel(self, fp2)

        if D is None:
            D = self.distance(self, fp2)

        if dD_dr1 is None:
            dD_dr1 = self.dD_drm(fp2, index1, D=D)

        if dD_dr2 is None:
            dD_dr2 = fp2.dD_drm(self, index2, D=D)

        prefactor = 1 / self.l**2 * kernel

        # Radial contribution:

        g1 = self.gradients[index1]
        g2 = fp2.gradients[index2]
        C1 = np.einsum('ijkl,ijkm->lm', g1, g2)

        # Angle contribution:

        g1 = self.anglegradients[index1]
        g2 = fp2.anglegradients[index2]
        C2 = np.einsum('ijklm,ijkln->mn', g1, g2)

        result = prefactor * (D**2 / self.l**2 *
                              np.outer(dD_dr1, dD_dr2) + C1 + C2)

        return result


class CartesianCoordFP(Fingerprint):

    def __init__(self, **kwargs):
        ''' Null fingerprint where the fingerprint vector is
        merely the flattened atomic coordinates. '''

        default_parameters = {'weight': 1.0,
                              'scale': 1.0}

        self.params = default_parameters.copy()
        self.params.update(kwargs)

        self.set_params()
        return

    def set_atoms(self, atoms):
        self.atoms = atoms
        self.set_params()

    def set_params(self):
        ''' Set parameters according to dictionary
            self.params '''

        self.l = self.params.get('scale')

        return

    def get_fingerprint_vector(self):
        return self.atoms.get_positions(wrap=False).reshape(-1)

    def update(self, params):
        if params is not None:
            for param in params:
                self.params[param] = params[param]
        self.set_params()
        return

    def calculate_gradient(self, index):
        gradient = np.zeros([len(self.atoms), 3])
        gradient[index, :] = 1.0
        return gradient.flatten()

    # ::: KERNEL STUFF ::: #
    # -------------------- #

    def distance(self, x1, x2):
        return pdist([x1.get_fingerprint_vector(),
                      x2.get_fingerprint_vector()])

    def kernel(self, x1, x2):
        return np.exp(-self.distance(x1, x2)**2 / 2 / self.l**2)

    def kernel_gradient(self, fp2, index, **kwargs):
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

    def dD_drm(self, fp2, index, D=None):

        if D is None:
            D = self.distance(self, fp2)

        if D == 0.0:
            return np.zeros(3)

        diffvec = self.get_fingerprint_vector() - fp2.get_fingerprint_vector()
        result = (diffvec / D)[index * 3: (index + 1) * 3]
        return result

    def kernel_hessian(self, fp2, index1, index2, **kwargs):

        prefactor = 1 / self.l**2 * self.kernel(self, fp2)

        if index1 == index2:
            C = np.eye(3)
        else:
            C = np.zeros([3, 3])

        diffvec = self.get_fingerprint_vector() - fp2.get_fingerprint_vector()
        diffvec1 = diffvec[index1 * 3: (index1 + 1) * 3]
        diffvec2 = -diffvec[index2 * 3: (index2 + 1) * 3]

        result = prefactor * (np.outer(diffvec1, diffvec2) / self.l**2 + C)

        return result


class ParallelOganovFP(OganovFP):

    def __init__(self, **kwargs):
        OganovFP.__init__(self, **kwargs)

    def calculate_gradient(self, index):
        '''
        Calculates the derivative of the fingerprint
        with respect to one of the coordinates.

        index: Atom index with which to differentiate
        '''
        gradient = np.zeros([self.n, self.n, self.N, 3])
        natoms = len(self.atoms)

        mask = np.arange(natoms) == index
        ext_mask = np.arange(len(self.extendedatoms)) % natoms == index

        # Sum over elements:
        for p in range(len(self.r_indices)):

            # Distribute:
            if p % world.size != world.rank:
                continue

            i, j = self.r_indices[p]
            indexi = mask[i]
            indexj = ext_mask[j]

            if not (indexi or indexj):
                continue

            if indexi and indexj:
                continue
            
            # position vector between atoms:
            rij = self.rm[p]
            Gij = self.Gij(p)

            if indexj:
                rij = -rij

            A, B = self.AB[p]
            gradient[A, B] += np.outer(Gij, -rij)

        # Gather all sub-gradients to a big container:
        gg = np.empty([world.size, self.n, self.n, self.N, 3])
        world.all_gather(gradient, gg)

        # Sum up:
        gradient = gg.sum(axis=0)

        if self.weight_by_elements:
            factortable = np.einsum('i,j->ij',
                                    self.elcounts,
                                    self.elcounts).astype(float)**-1
            gradient = np.einsum('ijkl,ij->ijkl', gradient, factortable)

        return gradient


class ParallelRadAngFP(RadialAngularFP, ParallelOganovFP):

    def __init__(self, **kwargs):
        RadialAngularFP.__init__(self, **kwargs)

    def set_angles(self):
        """
        In angle vector 'self.av' all angles are saved where
        one of the atoms is in 'self.atoms' and the other
        two are in 'self.extendedatoms'
        """

        # Extended distance and displacement vector matrices:
        ap = self.atoms.positions
        ep = self.extendedatoms.positions
        n0 = len(ep)

        # make sure ep can be scattered to world.size procs:
        lack = world.size - len(ep) % world.size

        # Pad with nans:
        ep_ext = np.concatenate((ep, np.full([lack, 3], np.nan)))
        assert len(ep_ext) % world.size == 0

        # Parallel calculation of self.edm:
        n = len(ep_ext)

        # Distribute positions to processors:
        mypart = np.zeros([int(n / world.size), 3])
        world.scatter(ep_ext, mypart, 0)

        # Calculate sub distance matrix:
        myedm = distance_matrix(mypart, ep_ext)

        # Collect data from processors:
        edm = np.empty([n, n])
        world.all_gather(myedm, edm)

        edm = edm[~np.isnan(edm)].reshape(n0, n0)

        n = len(ep)

        dm = distance_matrix(ap, ep)
        self.angleconstant = self.aweight / (pi / self.nanglebins)

        indices = []

        mask1 = np.logical_or(dm == 0, dm > self.Rtheta)
        mask2 = dm == 0
        mask3 = np.logical_or(edm == 0, edm > self.Rtheta)

        for i in range(len(self.atoms)):
            for j in range(len(self.extendedatoms)):

                if mask1[i, j]:
                    continue

                for k in range(len(self.extendedatoms)):

                    if mask2[i, k]:
                        continue

                    if mask3[j, k]:
                        continue

                    indices.append((i, j, k))

        self.indices = np.array(indices, dtype=int)

        self.arm = ap[self.indices[:, 0]] - ep[self.indices[:, 1]]
        self.adm = np.linalg.norm(self.arm, axis=1)
        self.erm = ep[self.indices[:, 2]] - ep[self.indices[:, 1]]
        self.edm = np.linalg.norm(self.erm, axis=1)

        args = np.einsum('ij,ij->i', self.arm, self.erm) / self.adm / self.edm

        # Take care of numerical errors:
        args = np.where(args >= 1.0, 1.0 - 1e-9, args)
        args = np.where(args <= -1.0, -1.0 + 1e-9, args)

        self.fcij = self.cutoff_function(self.adm)
        self.fcjk = self.cutoff_function(self.edm)
        self.thetas = np.arccos(args)

        self.ABC = [(self.prim_symbols[i], self.ext_symbols[j], self.ext_symbols[k])
                    for i, j, k in self.indices]

        return

    def calculate_angle_gradient(self, index, 
                                 firstvalues, secondvalues,
                                 third_i, third_j, third_k):

        '''
        Calculates the derivative of the fingerprint
        with respect to one of the coordinates.

        index: Atom index with which to differentiate
        '''
        gradient = np.zeros([self.n, self.n, self.n, self.nanglebins, 3])

        n = len(self.atoms)
        mask = np.arange(n) == index
        ext_mask = np.arange(len(self.extendedatoms)) % n == index

        # Distribute calculation:
        for p in range(len(self.indices)):

            if p % world.size != world.rank:
                continue

            i, j, k = self.indices[p]
            indexi = mask[i]
            indexj = ext_mask[j]
            indexk = ext_mask[k]

            if not (indexi or indexj or indexk):
                continue

            result = np.zeros([self.nanglebins, 3])
            if indexi:
                result += firstvalues[p] + third_i[p]
            if indexj:
                result += -firstvalues[p] + secondvalues[p] + third_j[p]
            if indexk:
                result += -secondvalues[p] + third_k[p]

            A, B, C = self.ABC[p]
            gradient[A, B, C] += result
            
        results = np.empty([world.size, self.n, self.n, self.n,
                            self.nanglebins, 3])
        world.all_gather(gradient, results)

        gradient = self.angleconstant * results.sum(axis=0)

        if self.weight_by_elements:
            factortable = np.einsum('i,j,k->ijk',
                                    self.elcounts,
                                    self.elcounts,
                                    self.elcounts).astype(float)**-1
            gradient = np.einsum('ijklm,ijk->ijklm', gradient, factortable)

        return gradient
