from scipy.spatial.distance import pdist
from scipy.signal import gaussian
from scipy.ndimage.interpolation import shift
from math import pi
from matplotlib import pyplot as plt
from scipy.spatial import distance_matrix
import numpy as np
import copy
import time
from warnings import catch_warnings, simplefilter

class OganovFP():


    def __init__(self, limit=10.0, delta=0.5, N=200, pbc=None):


        # constant parameters for fingerprint:
        self.limit = limit
        self.N = N
        self.pbc = pbc


        # modifiable parameters:
        self.params = {'weight': 1.0,
                       'scale': 1.0,
                       'delta': delta}

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
        self.n = len(self.elements)

        self.Nmat = np.ndarray([self.n, self.n])
        for i in range(self.Nmat.shape[0]):
            for j in range(self.Nmat.shape[1]):
                self.Nmat[i,j] = (len([atom for atom in self.atoms if 
                                       atom.symbol==self.elements[i]]) * 
                                  len([atom for atom in self.atoms if 
                                       atom.symbol==self.elements[j]]))

        self.extend_positions()
        self.set_element_matrix()
        self.update()


    def update(self, params=None):
        ''' Update method when parameters are changed '''

        if params is not None:
            for param in params:

                self.params[param] = params[param]

        self.set_params()
        self.set_peak_heights()
        self.get_fingerprint()
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

        # Number of cells needed to consider given the limit and pbc:
        ncells = [self.limit // lengths[i] + 1 for i in range(3)]
        nx, ny, nz = [1 + 2 * int(n) * self.pbc[i] for i,n in enumerate(ncells)]

        self.extendedatoms = self.atoms.repeat([nx, ny, nz])

        newstart = natoms * int(np.prod([nx, ny, nz]) / 2)
        newend = newstart + natoms
        self.atoms = self.extendedatoms[newstart:newend]

        # Distance matrix
        self.dm = distance_matrix(x=self.atoms.positions,
                                  y=self.extendedatoms.positions)

        # position vector matrix
        self.rm = np.ndarray([self.dm.shape[0], self.dm.shape[1], 3])
        for i in range(len(self.atoms)):
            for j in range(len(self.extendedatoms)):
                self.rm[i, j] = (self.atoms[i].position -
                                 self.extendedatoms[j].position)

        return


    def set_element_matrix(self):
        ''' Form the matrix for Ni and Nj used in calculating self.h '''

        ielements, icounts = np.unique(self.atoms.get_chemical_symbols(),
                                       return_counts=True)
        sortindices = np.argsort(ielements)
        ielements = ielements[sortindices]
        icounts = icounts[sortindices]

        f = lambda element: icounts[np.where(ielements == element)[0][0]]
        counti = [f(e) for e in self.atoms.get_chemical_symbols()]



        jelements, jcounts = np.unique(self.extendedatoms.get_chemical_symbols(),
                                       return_counts=True)
        sortindices = np.argsort(jelements)
        jelements = jelements[sortindices]
        jcounts = jcounts[sortindices]

        f = lambda element: jcounts[np.where(jelements == element)[0][0]]
        countj = [f(e) for e in self.extendedatoms.get_chemical_symbols()]

        self.em = np.outer(counti, countj).astype(int)

        return self.em


    def set_peak_heights(self):
        ''' Calculate the delta peak heights self.h '''

        # Ignore zero-division warning
        with catch_warnings():
            simplefilter("ignore", category=RuntimeWarning)
            self.h = np.where(self.dm > 0.0001,
                              (self.Vuc / (4 * pi * self.dm**2 *
                                           self.em *
                                           self.delta)),
                              0.0)

        return self.h


    def get_fingerprint(self):
        ''' Calculate the Gaussian-broadened fingerprint. '''

        self.G = np.ndarray([self.n, self.n, self.N])
        x = np.linspace(0, self.limit, self.N) # variable array

        # Broadening of each peak:
        for i in range(self.n):
            for j in range(self.n):

                R, h = [], []

                # Get peak positions
                R = np.where(self.dm < self.limit,
                             self.dm,
                             0.0)

                # Get peak heights
                h = np.where(self.dm < self.limit,
                             self.h,
                             0.0)

                # Consider only the correct elements i and j
                ms = np.array(self.atoms.get_chemical_symbols()) == self.elements[i]
                ns = np.array(self.extendedatoms.get_chemical_symbols()) == self.elements[j]

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

                npeaks = len(R) # number of peaks
                
                g = np.zeros(self.N)
                for p in range(npeaks):
                    g += h[p] * np.exp(- (x - R[p])**2 / 2 / self.delta**2)

                self.G[i,j] = g
                

        return self.G
        

    def get_fingerprint_vector(self):
        return self.G.flatten()


    ### ::: GRADIENTS ::: ###
    ### ----------------- ###

    def calculate_gradient(self, index):
        '''
        Calculates the derivative of the fingerprint with respect to one of the coordinates.

        index: Atom index with which to differentiate
        '''
        gradient = np.zeros([self.n, self.N, 3])

        i = index
        A = list(self.elements).index(self.atoms[i].symbol)

        elementlist = list(self.elements)
        jsymbols = [elementlist.index(atom.symbol) for atom in self.extendedatoms]
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


        return gradient


    def calculate_all_gradients(self):

        self.gradients = np.array([self.calculate_gradient(atom.index) for atom in self.atoms])
        return self.gradients



    ### ::: KERNEL STUFF ::: ###
    ### -------------------- ###


    def distance(self, x1, x2):
        return pdist([x1.get_fingerprint_vector(),
                      x2.get_fingerprint_vector()])


    def kernel(self, x1, x2):

        # return (self.weight**2
        #         * np.exp(-self.distance(x1, x2)**2 / 2 / self.l**2))

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
                                                     axes=[0,0])

        result = Bsum / D
        
        return result
    

    def kernel_hessian(self, fp2, index1, index2):
        
        prefactor = 1 / self.l**2 * self.kernel(self, fp2)

        g1 = self.gradients[index1]
        g2 = fp2.gradients[index2]

        A1 = list(self.elements).index(self.atoms[index1].symbol)
        A2 = list(fp2.elements).index(fp2.atoms[index2].symbol)

        tildexvec = self.G - fp2.G

        Qm = np.zeros(3)
        Qn = np.zeros(3)
        for B in range(self.n):
            Qm +=   (1 + int(B != A1)) * np.tensordot(tildexvec[B, A1], g1[B], axes=[0,0])
            Qn += - (1 + int(B != A2)) * np.tensordot(tildexvec[B, A2], g2[B], axes=[0,0])
            
        C = np.zeros([3,3])
        for B in range(self.n): # sum over elements

            if A1 == A2:
                C += (1 + int(B != A2)) * np.tensordot(g1[B], g2[B], axes=[0, 0])

            else:
                if B in [A1, A2]:
                    C += np.tensordot(g1[A2], g2[A1], axes=[0, 0])

        result = prefactor * (1 / self.l**2 * np.outer(Qm, Qn) + C)

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
                first = self.G[A,B] - fp2.G[A,B]
                second = dFP_dDelta1[A,B] - dFP_dDelta2[A,B]
                result += first.dot(second)

        result *= 1 / D
                
        return result


    def dFP_dDelta(self):

        if self.dFP_dDelta_calculated:
            return self.dfp_ddelta
        
        xvec = np.linspace(0., self.limit, self.N)
        result = np.zeros([self.n, self.n, self.N])

        elementlist = list(self.elements)
        isymbols = [elementlist.index(atom.symbol) for atom in self.atoms]
        jsymbols = [elementlist.index(atom.symbol) for atom in self.extendedatoms]
        
        for A in range(self.n):
            for B in range(A, self.n):

                for i in range(len(self.atoms)):
                    if A != isymbols[i]:
                        continue
                    
                    for j in range(len(self.extendedatoms)):
                        if B != jsymbols[j]:
                            continue

                        xij = self.dm[i,j]

                        if xij == 0 or xij > self.limit:
                            continue

                        normsq = (xvec - xij)**2
                        subresult = np.exp(- normsq / 2 / self.delta**2)
                        subresult *= (normsq / self.delta**2 - 1) * self.h[i,j]

                        result[A,B] += subresult
                result[B,A] = result[A,B] # symmetric matrix
                        
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
            tildexvec = dFP_dDelta1[A,B] - dFP_dDelta2[A,B]
            jsum = self.gradients[index][B]
            Bsum += (1 + int(A != B)) * np.tensordot(tildexvec, jsum, axes=[0,0])

        second = prefactor * Bsum


        elementlist = list(self.elements)
        jsymbols = [elementlist.index(atom.symbol) for atom in self.extendedatoms]
        Bsum = np.zeros(3)
        # Sum over elements:
        for B in range(self.n):
            tildexvec = self.G[A,B] - fp2.G[A,B]
            jsum = np.zeros([self.N, 3])

            for j in range(len(self.extendedatoms)):

                if B != jsymbols[j]:
                    continue

                rij = self.rm[i,j] #self.atoms[i].position - self.extendedatoms[j].position
                jsum += (1 + int(A == B)) * np.outer(self.dGij_dDelta(i,j), -rij)

            Bsum += (1 + int(A != B)) * np.tensordot(tildexvec, jsum, axes=[0,0])        

        third = prefactor * Bsum


        return first + second + third


    def Gij(self, i, j):
        xij = self.dm[i,j]

        if xij == 0 or xij > self.limit:
            return 0.0

        xvec = np.linspace(0., self.limit, self.N)
        diffvec = xvec - xij
        Gij = ((2 / xij - diffvec / self.delta**2) * 
               self.h[i,j] / xij * 
               np.exp(- diffvec**2 / 2 / self.delta**2))

        return Gij

        
    def dGij_dDelta(self, i, j):

        xvec = np.linspace(0., self.limit, self.N)
        xij = self.dm[i,j]

        if xij == 0 or xij > self.limit:
            return 0

        normsq = (xvec - xij)**2

        first = 1 / self.delta * self.h[i,j] / xij * np.exp(-normsq / 2 / self.delta**2)
        second = 2 / xij * (normsq / self.delta**2 - 1) + (xvec - xij) / self.delta**2 * (3 - normsq / self.delta**2)

        return first * second

    # ----------------------------------------------
    # d_dDelta: Hessian


    def dk_drm_drn_dDelta(self, fp2, index1, index2):

        t0 = time.time()
        dD_dDelta = self.dD_dDelta(fp2)
        D = self.distance(self, fp2)
        first = - D / self.l**2 * dD_dDelta * self.kernel_hessian(fp2,
                                                                  index1,
                                                                  index2)

        # t1 = time.time()
        prefactor = 1 / self.l**2 * self.kernel(self, fp2)

        second = D * np.outer(self.d_dDelta_D_dD_drm(fp2, index1),
                               fp2.dD_drm(self, index2))
        second += D * np.outer(self.dD_drm(fp2, index1),
                               fp2.d_dDelta_D_dD_drm(self, index2))
        second *= prefactor / self.l**2

        # t2 = time.time()

        third = np.zeros([3,3])
        A1 = list(self.elements).index(self.atoms[index1].symbol)
        A2 = list(fp2.elements).index(fp2.atoms[index2].symbol)
        d1 = self.d_dDelta_dFP_drm(index1)
        d2 = fp2.d_dDelta_dFP_drm(index2)
        
        for B in range(self.n):
            if A1 == A2:
                prefactor2 = (1 + int(A1 != B))
                third += prefactor2 * np.tensordot(d1[B],
                                                   fp2.gradients[index2][B],
                                                   axes=[0,0])
                third += prefactor2 * np.tensordot(self.gradients[index1][B],
                                                   d2[B],
                                                   axes=[0,0])
            else:
                if B in [A1, A2]:
                    third += np.tensordot(d1[A2],
                                          fp2.gradients[index2][A1],
                                          axes=[0,0])
                    third += np.tensordot(self.gradients[index1][A2],
                                          d2[A1],
                                          axes=[0,0])
        third *= prefactor
        t3 = time.time()
        #print(1, t1-t0)
        #print(2, t2-t1)
        # print("Time in dk_drm_drn_dDelta: %.04f sec" % (t3-t0))

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
            tildexvec = self.G[A,B] - fp2.G[A,B]
            tildexvec_dDelta = dFP_dDelta1[A,B] - dFP_dDelta2[A,B]
            prefactor = 1 + int(A != B)
            Bsum += prefactor * np.tensordot(tildexvec_dDelta, g[B], axes=[0,0])
            Bsum += prefactor * np.tensordot(tildexvec, d2[B], axes=[0,0])
            
        return Bsum


    def d_dDelta_dFP_drm(self, index):

        # i = index
        # A = list(self.elements).index(self.atoms[i].symbol)
        # result = np.zeros([self.n, self.N, 3])

        # elementlist = list(self.elements)
        # jsymbols = np.array([elementlist.index(atom.symbol) for atom in self.extendedatoms])
        # factors = np.ones([self.n, self.n]) + np.eye(self.n)

        # for B in range(self.n):

        #     jsum = np.zeros([self.N, 3])

        #     for j in range(len(self.extendedatoms)):

        #         if B != jsymbols[j]:
        #            continue

        #         jsum += np.outer(self.dGij_dDelta(i,j), -self.rm[i,j])

        #     result[B] += factors[A,B] * jsum
        
        # return result



        # Precalculate:
        
        if self.d_dDelta_dFP_drm_calculated:
            return self.d_ddelta_dfp_drm[index]

        self.d_ddelta_dfp_drm = np.zeros([len(self.atoms), self.n, self.N, 3])
        elementlist = list(self.elements)
        isymbols = np.array([elementlist.index(atom.symbol) for atom in self.atoms])
        jsymbols = np.array([elementlist.index(atom.symbol) for atom in self.extendedatoms])
        factors = np.ones([self.n, self.n]) + np.eye(self.n)

        for i in range(len(self.atoms)):

            A = isymbols[i]

            for B in range(self.n):

                jsum = np.zeros([self.N, 3])

                for j in range(len(self.extendedatoms)):

                    if B != jsymbols[j]:
                       continue

                    jsum += np.outer(self.dGij_dDelta(i,j), -self.rm[i,j])

                self.d_ddelta_dfp_drm[i][B] += factors[A,B] * jsum
        
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
        return self.kernel(self, fp2) * 2 # / self.weight
        
                 
class CartesianCoordinatesFP():
    
    def __init__(self):

        self.params = {'weight': 1.0,
                       'scale': 1.0}

        # In order to get the units right while
        # training a Gaussian process, we need
        # to specify the parameter that determines
        # the lengthscale of the problem:
        self.lengthscaleparam = 'scale'

        return


    def set_params(self):
        self.weight = self.params['weight']
        self.l = self.params['scale']
        self.lengthscale = self.params[self.lengthscaleparam]
        return
    

    def set_atoms(self, atoms):
        self.atoms = atoms


    def get_fingerprint_vector(self):
        return self.atoms.positions.flatten()


    def update(self, params={}):
        for param in params:
            self.params[param] = params[param]

        self.set_params()
        return
    

    def distance(self, x1, x2):
        return pdist([x1.get_fingerprint_vector(),
                      x2.get_fingerprint_vector()])


    def kernel(self, x1, x2):
        assert x1.l == x2.l
        return np.exp(-self.distance(x1, x2)**2 / 2 / self.l**2)

    
    def kernel_gradient(self, fp2, index):
        """
        Calculates the derivative of the kernel between
        self and fp2 with respect to atom with index 'index' in atom set
        of self.
        """

        x1 = self.atoms[index].position
        x2 = fp2.atoms[index].position
        prefactor = -(x1 - x2) / self.l**2
        return prefactor * self.kernel(self, fp2)


    def kernel_hessian(self, fp2, index1, index2):

        x1 = self.atoms[index1].position - fp2.atoms[index1].position
        x2 = self.atoms[index2].position - fp2.atoms[index2].position 
        P = np.outer(x1, x2) / self.l**2
        
        prefactor = (np.identity(3) * int(index1 == index2) - P) / self.l**2

        assert prefactor.shape[0] == 3
        assert prefactor.shape[1] == 3

        return prefactor * self.kernel(self, fp2)

