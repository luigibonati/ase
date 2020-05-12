""" Maximally localized Wannier Functions

    Find the set of maximally localized Wannier functions
    using the spread functional of Marzari and Vanderbilt
    (PRB 56, 1997 page 12847).
"""
import os
import warnings
import functools
from time import time
from math import sqrt, pi
from pickle import dump, load
from scipy.special import erf
from scipy.linalg import qr

import numpy as np

from ase.parallel import paropen
from ase.dft.kpoints import get_monkhorst_pack_size_and_offset
from ase.transport.tools import dagger, normalize

dag = dagger


def gram_schmidt(U):
    """Orthonormalize columns of U according to the Gram-Schmidt procedure."""
    for i, col in enumerate(U.T):
        for col2 in U.T[:i]:
            col -= col2 * np.dot(col2.conj(), col)
        col /= np.linalg.norm(col)


def gram_schmidt_single(U, n):
    """Orthogonalize columns of U to column n"""
    N = len(U.T)
    v_n = U.T[n]
    indices = list(range(N))
    del indices[indices.index(n)]
    for i in indices:
        v_i = U.T[i]
        v_i -= v_n * np.dot(v_n.conj(), v_i)


def lowdin(U, S=None):
    """Orthonormalize columns of U according to the symmetric Lowdin procedure.
       The implementation uses SVD, like symm. Lowdin it returns the nearest
       orthonormal matrix, but is more robust.
    """

    L, s, R = np.linalg.svd(U, full_matrices=False)
    U[:] = np.dot(L, R)


def neighbor_k_search(k_c, G_c, kpt_kc, tol=1e-4):
    # search for k1 (in kpt_kc) and k0 (in alldir), such that
    # k1 - k - G + k0 = 0
    alldir_dc = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
                          [1, 1, 0], [1, 0, 1], [0, 1, 1]], int)
    for k0_c in alldir_dc:
        for k1, k1_c in enumerate(kpt_kc):
            if np.linalg.norm(k1_c - k_c - G_c + k0_c) < tol:
                return k1, k0_c

    print('Wannier: Did not find matching kpoint for kpt=', k_c)
    print('Probably non-uniform k-point grid')
    raise NotImplementedError


def calculate_weights(cell_cc):
    """ Weights are used for non-cubic cells, see PRB **61**, 10040"""
    alldirs_dc = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1],
                           [1, 1, 0], [1, 0, 1], [0, 1, 1]], dtype=int)
    g = np.dot(cell_cc, cell_cc.T)
    # NOTE: Only first 3 of following 6 weights are presently used:
    w = np.zeros(6)
    w[0] = g[0, 0] - g[0, 1] - g[0, 2]
    w[1] = g[1, 1] - g[0, 1] - g[1, 2]
    w[2] = g[2, 2] - g[0, 2] - g[1, 2]
    w[3] = g[0, 1]
    w[4] = g[0, 2]
    w[5] = g[1, 2]
    # Make sure that first 3 Gdir vectors are included -
    # these are used to calculate Wanniercenters.
    Gdir_dc = alldirs_dc[:3]
    weight_d = w[:3]
    for d in range(3, 6):
        if abs(w[d]) > 1e-5:
            Gdir_dc = np.concatenate((Gdir_dc, alldirs_dc[d:d + 1]))
            weight_d = np.concatenate((weight_d, w[d:d + 1]))
    weight_d /= max(abs(weight_d))
    return weight_d, Gdir_dc


def random_orthogonal_matrix(dim, seed=None, real=False):
    """Generate a random orthogonal matrix"""
    from scipy.stats import special_ortho_group
    if real:
        ortho_m = special_ortho_group.rvs(dim=dim, random_state=seed)
    else:
        ortho_m = special_ortho_group.rvs(
            dim=dim, random_state=seed).astype(np.complex128)
    return ortho_m


# def steepest_descent(func, step=.005, tolerance=1e-6, **kwargs):
def steepest_descent(func, step=.005, tolerance=1e-6, verbose=False, **kwargs):
    fvalueold = 0.
    fvalue = fvalueold + 10
    count = 0
    while abs((fvalue - fvalueold) / fvalue) > tolerance:
        fvalueold = fvalue
        dF = func.get_gradients()
        func.step(dF * step, **kwargs)
        fvalue = func.get_functional_value()
        count += 1
        if verbose:
            print('SteepestDescent: iter=%s, value=%s' % (count, fvalue))


def md_min(func, step=.25, tolerance=1e-6, max_iter=10000,
           verbose=False, **kwargs):
    if verbose:
        print('Localize with step =', step, 'and tolerance =', tolerance)
        finit = func.get_functional_value()
    t = -time()
    fvalueold = 0.
    fvalue = fvalueold + 10
    count = 0
    V = np.zeros(func.get_gradients().shape, dtype=complex)

    while abs((fvalue - fvalueold) / fvalue) > tolerance:
        fvalueold = fvalue
        dF = func.get_gradients()

        V *= (dF * V.conj()).real > 0
        V += step * dF
        func.step(V, **kwargs)
        fvalue = func.get_functional_value()

        if fvalue < fvalueold:
            step *= 0.5
        count += 1
        if verbose:
            print('MDmin: iter=%s, step=%s, value=%0.4f'
                  % (count, step, fvalue))
        if count > max_iter:
            t += time()
            warnings.warn('Max iterations reached: '
                          'iters=%s, step=%s, seconds=%0.2f, value=%0.4f'
                          % (count, step, t, fvalue))
            break
    if verbose:
        t += time()
        print('%d iterations in %0.2f seconds (%0.2f ms/iter), endstep = %s' %
              (count, t, t * 1000. / count, step))
        print('Initial value=%0.4f, Final value=%0.4f' %
              (finit, fvalue))


def rotation_from_projection2(proj_nw, fixed, ortho=True):
    V_ni = proj_nw
    Nb, Nw = proj_nw.shape
    M = fixed
    L = Nw - M
    U_ww = np.zeros((Nw, Nw), dtype=proj_nw.dtype)
    c_ul = np.zeros((Nb - M, L), dtype=proj_nw.dtype)
    for V_n in V_ni.T:
        V_n /= np.linalg.norm(V_n)

    # Find EDF
    P_ui = V_ni[M:].copy()
    la = np.linalg
    for l in range(L):
        norm_list = np.array([la.norm(v) for v in P_ui.T])
        perm_list = np.argsort(-norm_list)
        P_ui = P_ui[:, perm_list].copy()    # largest norm to the left
        P_ui[:, 0] /= la.norm(P_ui[:, 0])   # normalize
        c_ul[:, l] = P_ui[:, 0]             # save normalized EDF
        gram_schmidt_single(P_ui, 0)        # ortho remain. to this EDF
        P_ui = P_ui[:, 1:].copy()           # remove this EDF

    U_ww[:M] = V_ni[:M, :]
    U_ww[M:] = np.dot(c_ul.T.conj(), V_ni[M:])

    if ortho:
        gram_schmidt(U_ww)
    else:
        normalize(U_ww)

    return U_ww, c_ul


def rotation_from_projection(proj_nw, fixed, ortho=True):
    """Determine rotation and coefficient matrices from projections

    proj_nw = <psi_n|p_w>
    psi_n: eigenstates
    p_w: localized function

    Nb (n) = Number of bands
    Nw (w) = Number of wannier functions
    M  (f) = Number of fixed states
    L  (l) = Number of extra degrees of freedom
    U  (u) = Number of non-fixed states
    """

    Nb, Nw = proj_nw.shape
    M = fixed
    L = Nw - M
    U = Nb - M

    U_ww = np.empty((Nw, Nw), dtype=proj_nw.dtype)
    U_ww[:M] = proj_nw[:M]

    # choose method between ['std', 'svd', 'qrcp']
    # it could become an argument for the function
    # NOTE: for now this is a dirty workaround not to touch the GPAW code and
    #  still be able to change the projection method from a job script.
    method = 'std'
    if 'ASE_WAN_PROJ' in os.environ:
        env = os.environ['ASE_WAN_PROJ']
        if env == 'qrcp':
            method = env
        elif env == 'svd':
            method = env

    # If there are extra degrees of freedom we have to select L of them
    if L > 0:
        C_ul = np.empty((U, L), dtype=proj_nw.dtype)
        if method == 'std':
            # Unknown method, very similar to SVD
            #  but the results slightly differ
            proj_uw = proj_nw[M:]
            eig_w, C_ww = np.linalg.eigh(np.dot(dag(proj_uw), proj_uw))
            C_ul = np.dot(proj_uw, C_ww[:, np.argsort(-eig_w.real)[:L]])
            # eig_u, C_uu = np.linalg.eigh(np.dot(proj_uw, dag(proj_uw)))
            # C_ul = C_uu[:, np.argsort(-eig_u.real)[:L]]

            U_ww[M:] = np.dot(dag(C_ul), proj_uw)

        elif method == 'svd':
            # SVD implementation of nearest orthogonal matrix, aka symm. Lowdin
            proj_uw = proj_nw[M:]
            C_ww = np.dot(dag(proj_uw), proj_uw)
            Ls, s, Rs = np.linalg.svd(C_ww, full_matrices=False)
            C_ww = np.dot(Ls, Rs)

            # Keep only the L columns with the biggest singular values
            C_ul = np.empty((U, L), dtype=proj_uw.dtype)
            C_ul = np.dot(proj_uw, C_ww[:, :L])

            U_ww[M:] = np.dot(dag(C_ul), proj_uw)

        elif method == 'qrcp':
            # QRCP implemetation, keep the columns based on the permutation
            proj_uw = proj_nw[M:]
            C_ww = np.dot(dag(proj_uw), proj_uw)
            Q, R, P = qr(C_ww, mode='full', pivoting=True, check_finite=True)
            C_ww = C_ww[:, P]

            # Keep only the L columns with the biggest diagonal elements on R
            C_ul = np.dot(proj_uw, C_ww[:, :L])
            lowdin(C_ul)

            U_ww[M:] = np.dot(dag(C_ul), proj_uw)

        normalize(C_ul)
    else:
        C_ul = np.empty((U, 0), dtype=proj_nw.dtype)

    if ortho:
        # gram_schmidt(U_ww)
        lowdin(U_ww)
    else:
        normalize(U_ww)

    return U_ww, C_ul


def scdm(calc, Nw, fixed_k, verbose=True):
    """Compute localized orbitals with SCDM method

    This method was published by Anil Damle and Lin Lin in Multiscale
    Modeling & Simulation 16, 1392–1410 (2018).
    For now only the isolated bands algorithm is implemented, because it is
    intended as a drop-in replacement for other initial guess methods for
    the ASE Wannier class.

    calc   = GPAW calculator object
    Nk (k) = Number of k-points
    Nb (n) = Number of bands
    Nw (w) = Number of wannier functions
    M  (f) = Number of fixed states (fixed_k)
    L  (l) = Number of extra degrees of freedom
    U  (u) = Number of non-fixed states
    """

    Nb = calc.get_number_of_bands()
    kpts = calc.get_bz_k_points()
    gamma_idx = [i.all() for i in np.isclose(kpts, 0, atol=1e-10)].index(True)
    Nk = len(kpts)
    U_kww = []
    C_kul = []

    # get grid size and check that we have Nw bands at once
    Ng = np.prod(calc.wfs.gd.N_c)
    ps_wfs = np.zeros((Nb, Ng), dtype=np.complex128)

    # compute factorization just at Gamma point
    for n in range(Nb):
        ps_wfs[n] = calc.get_pseudo_wave_function(band=n, kpt=gamma_idx,
                                                  spin=0).ravel()

    _, _, P = qr(ps_wfs, mode='full', pivoting=True, check_finite=True)

    for k in range(Nk):
        for n in range(Nb):
            ps_wfs[n] = calc.get_pseudo_wave_function(band=n, kpt=k,
                                                      spin=0).ravel()

        A_nw = ps_wfs[:, P[:Nw]]
        U_ww, C_ul = rotation_from_projection(proj_nw=A_nw,
                                              fixed=fixed_k[k],
                                              ortho=True)
        U_kww.append(U_ww)
        C_kul.append(C_ul)

    U_kww = np.asarray(U_kww)

    return C_kul, U_kww


def init_orbitals(calc, ntot, seed=None):
    """Place d-orbitals for every atom that has some in the valence states
        and then random s-orbitals close to the other atoms (< 1.5Å).
       'calc': ASE calculator object
       'ntot': total number of needed orbitals
       'seed': seed for random numbers"""

    atoms = calc.get_atoms()
    orbs = []
    No = 0
    for i, z in enumerate(atoms.get_atomic_numbers()):
        # check occupied d-orbitals in valence states in the GPAW setup
        s = calc.setups[i]
        if 2 in s.l_j and s.f_j[s.l_j.index(2)] > 0:
            No_new = No + 5
            if No_new <= ntot:
                orbs.append([i, 2, 1])
                No = No_new

    if No < ntot:
        # add random s-like orbitals if there are not enough yet
        if seed is not None:
            np.random.seed(seed)

        Ns = ntot - No
        tmp_atoms = atoms.copy()
        tmp_atoms.append('H')
        s_pos = tmp_atoms.get_scaled_positions()
        for i in range(0, Ns):
            fine = False
            while not fine:
                x, y, z = np.random.random(3)
                s_pos[-1] = [x, y, z]
                tmp_atoms.set_scaled_positions(s_pos)
                dists = tmp_atoms.get_distances(
                    a=-1,
                    indices=range(atoms.get_global_number_of_atoms()))
                if (dists < 1.5).any():
                    fine = True
            orbs.append([[x, y, z], 0, 1])

    assert sum([orb[1] * 2 + 1 for orb in orbs]) == ntot
    return orbs


class Wannier:
    """Maximally localized Wannier Functions

    Find the set of maximally localized Wannier functions using the
    spread functional of Marzari and Vanderbilt (PRB 56, 1997 page
    12847).
    """

    def __init__(self, nwannier, calc,
                 file=None,
                 nbands=None,
                 fixedenergy=None,
                 fixedstates=None,
                 spin=0,
                 initialwannier='orbitals',
                 functional='std',
                 seed=None,
                 verbose=False):
        """
        Required arguments:

          ``nwannier``: The number of Wannier functions you wish to construct.
            This must be at least half the number of electrons in the system
            and at most equal to the number of bands in the calculation.

          ``calc``: A converged DFT calculator class.
            If ``file`` arg. is not provided, the calculator *must* provide the
            method ``get_wannier_localization_matrix``, and contain the
            wavefunctions (save files with only the density is not enough).
            If the localization matrix is read from file, this is not needed,
            unless ``get_function`` or ``write_cube`` is called.

        Optional arguments:

          ``nbands``: Bands to include in localization.
            The number of bands considered by Wannier can be smaller than the
            number of bands in the calculator. This is useful if the highest
            bands of the DFT calculation are not well converged.

          ``spin``: The spin channel to be considered.
            The Wannier code treats each spin channel independently.

          ``fixedenergy`` / ``fixedstates``: Fixed part of Heilbert space.
            Determine the fixed part of Hilbert space by either a maximal
            energy *or* a number of bands (possibly a list for multiple
            k-points).
            Default is None meaning that the number of fixed states is equated
            to ``nwannier``.

          ``file``: Read localization and rotation matrices from this file.

          ``initialwannier``: Initial guess for Wannier rotation matrix.
            Can be 'bloch' to start from the Bloch states, 'random' to be
            randomized, 'gaussians' to start from randomly placed gaussian
            centers, 'orbitals' to start from atom-centered d-orbitals and
            randomly placed gaussian centers (see init_orbitals()),
            'scdm' to start from localized state selected with SCDM
            or a list passed to calc.get_initial_wannier.

          ``functional``: The functional used to measure the localization.
            Can be 'std' for the standard quadratic functional from the PRB
            paper, 'sigmoid' for a translated and rescaled sigmoid function,
            'erf' for a scaled error function, 'sqrt' for a square root
            function, 'cbrt' for cube root function.
            Every additional function is applied to the standard functional
            value.

          ``seed``: Seed for 'random' or 'gaussians' ``initialwannier``.

          ``verbose``: True / False level of verbosity.
          """
        # Bloch phase sign convention.
        # May require special cases depending on which code is used.
        sign = -1

        self.nwannier = nwannier
        self.calc = calc
        self.spin = spin
        self.functional = functional
        if verbose:
            print('Using functional:', functional)
        self.verbose = verbose
        self.kpt_kc = calc.get_bz_k_points()
        assert len(calc.get_ibz_k_points()) == len(self.kpt_kc)
        self.kptgrid = get_monkhorst_pack_size_and_offset(self.kpt_kc)[0]
        self.kpt_kc *= sign

        self.Nk = len(self.kpt_kc)
        self.unitcell_cc = calc.get_atoms().get_cell()
        self.largeunitcell_cc = (self.unitcell_cc.T * self.kptgrid).T
        self.weight_d, self.Gdir_dc = calculate_weights(self.largeunitcell_cc)
        self.Ndir = len(self.weight_d)  # Number of directions

        if nbands is not None:
            self.nbands = nbands
        else:
            self.nbands = calc.get_number_of_bands()
        if fixedenergy is None:
            if fixedstates is None:
                self.fixedstates_k = np.array([nwannier] * self.Nk, int)
            else:
                if isinstance(fixedstates, int):
                    fixedstates = [fixedstates] * self.Nk
                self.fixedstates_k = np.array(fixedstates, int)
        else:
            # Setting number of fixed states and EDF from specified energy.
            # All states below this energy (relative to Fermi level) are fixed.
            fixedenergy += calc.get_fermi_level()
            # print(fixedenergy)
            self.fixedstates_k = np.array(
                [calc.get_eigenvalues(k, spin).searchsorted(fixedenergy)
                 for k in range(self.Nk)], int)
        self.edf_k = self.nwannier - self.fixedstates_k
        if verbose:
            print('Wannier: Fixed states            : %s' % self.fixedstates_k)
            print('Wannier: Extra degrees of freedom: %s' % self.edf_k)

        # Set the list of neighboring k-points k1, and the "wrapping" k0,
        # such that k1 - k - G + k0 = 0
        #
        # Example: kpoints = (-0.375,-0.125,0.125,0.375), dir=0
        # G = [0.25,0,0]
        # k=0.375, k1= -0.375 : -0.375-0.375-0.25 => k0=[1,0,0]
        #
        # For a gamma point calculation k1 = k = 0,  k0 = [1,0,0] for dir=0
        if self.Nk == 1:
            self.kklst_dk = np.zeros((self.Ndir, 1), int)
            k0_dkc = self.Gdir_dc.reshape(-1, 1, 3)
        else:
            self.kklst_dk = np.empty((self.Ndir, self.Nk), int)
            k0_dkc = np.empty((self.Ndir, self.Nk, 3), int)

            # Distance between kpoints
            kdist_c = np.empty(3)
            for c in range(3):
                # make a sorted list of the kpoint values in this direction
                slist = np.argsort(self.kpt_kc[:, c], kind='mergesort')
                skpoints_kc = np.take(self.kpt_kc, slist, axis=0)
                kdist_c[c] = max([skpoints_kc[n + 1, c] - skpoints_kc[n, c]
                                  for n in range(self.Nk - 1)])

            for d, Gdir_c in enumerate(self.Gdir_dc):
                for k, k_c in enumerate(self.kpt_kc):
                    # setup dist vector to next kpoint
                    G_c = np.where(Gdir_c > 0, kdist_c, 0)
                    if max(G_c) < 1e-4:
                        self.kklst_dk[d, k] = k
                        k0_dkc[d, k] = Gdir_c
                    else:
                        self.kklst_dk[d, k], k0_dkc[d, k] = \
                            neighbor_k_search(k_c, G_c, self.kpt_kc)

        # Set the inverse list of neighboring k-points
        self.invkklst_dk = np.empty((self.Ndir, self.Nk), int)
        for d in range(self.Ndir):
            for k1 in range(self.Nk):
                self.invkklst_dk[d, k1] = self.kklst_dk[d].tolist().index(k1)

        Nw = self.nwannier
        Nb = self.nbands
        self.Z_dkww = np.empty((self.Ndir, self.Nk, Nw, Nw), complex)
        self.V_knw = np.zeros((self.Nk, Nb, Nw), complex)
        if file is None:
            self.Z_dknn = np.empty((self.Ndir, self.Nk, Nb, Nb), complex)
            for d, dirG in enumerate(self.Gdir_dc):
                for k in range(self.Nk):
                    k1 = self.kklst_dk[d, k]
                    k0_c = k0_dkc[d, k]
                    self.Z_dknn[d, k] = calc.get_wannier_localization_matrix(
                        nbands=Nb, dirG=dirG, kpoint=k, nextkpoint=k1,
                        G_I=k0_c, spin=self.spin)
        self.initialize(file=file, initialwannier=initialwannier, seed=seed)

    def initialize(self, file=None, initialwannier='orbitals', seed=None):
        """Re-initialize current rotation matrix.

        Keywords are identical to those of the constructor.
        """
        Nw = self.nwannier
        Nb = self.nbands

        if file is not None:
            self.Z_dknn, self.U_kww, self.C_kul = load(paropen(file, 'rb'))
        elif initialwannier == 'bloch':
            # Set U and C to pick the lowest Bloch states
            self.U_kww = np.zeros((self.Nk, Nw, Nw), complex)
            self.C_kul = []
            for U, M, L in zip(self.U_kww, self.fixedstates_k, self.edf_k):
                U[:] = np.identity(Nw, complex)
                if L > 0:
                    self.C_kul.append(
                        np.identity(Nb - M, complex)[:, :L])
                else:
                    self.C_kul.append([])
        elif initialwannier == 'random':
            # Set U and C to random (orthogonal) matrices
            self.U_kww = np.zeros((self.Nk, Nw, Nw), complex)
            self.C_kul = []
            for U, M, L in zip(self.U_kww, self.fixedstates_k, self.edf_k):
                U[:] = random_orthogonal_matrix(Nw, seed, real=False)
                if L > 0:
                    self.C_kul.append(random_orthogonal_matrix(
                        Nb - M, seed=seed, real=False)[:, :L])
                else:
                    self.C_kul.append(np.array([]))
        elif initialwannier == 'gaussians':
            # Create gaussians in random positions
            if seed is not None:
                np.random.seed(seed)
            centers = []
            for i in range(self.nwannier):
                centers.append(([np.random.random(),
                                 np.random.random(),
                                 np.random.random()], 0, 1))
            self.C_kul, self.U_kww = self.calc.initial_wannier(
                centers, self.kptgrid, self.fixedstates_k,
                self.edf_k, self.spin, self.nbands)
        elif initialwannier == 'orbitals':
            self.C_kul, self.U_kww = self.calc.initial_wannier(
                init_orbitals(self.calc, self.nwannier, seed),
                self.kptgrid, self.fixedstates_k,
                self.edf_k, self.spin, self.nbands)
        elif initialwannier == 'scdm':
            self.C_kul, self.U_kww = scdm(calc=self.calc,
                                          Nw=self.nwannier,
                                          fixed_k=self.fixedstates_k,
                                          verbose=self.verbose)
        else:
            # Use initial guess to determine U and C
            self.C_kul, self.U_kww = self.calc.initial_wannier(
                initialwannier, self.kptgrid, self.fixedstates_k,
                self.edf_k, self.spin, self.nbands)
        self.update()

    def save(self, file):
        """Save information on localization and rotation matrices to file."""
        dump((self.Z_dknn, self.U_kww, self.C_kul), paropen(file, 'wb'))

    def update(self):
        # Update large rotation matrix V (from rotation U and coeff C)
        for k, M in enumerate(self.fixedstates_k):
            self.V_knw[k, :M] = self.U_kww[k, :M]
            if M < self.nwannier:
                self.V_knw[k, M:] = np.dot(self.C_kul[k], self.U_kww[k, M:])
            # else: self.V_knw[k, M:] = 0.0

        # Calculate the Zk matrix from the large rotation matrix:
        # Zk = V^d[k] Zbloch V[k1]
        for d in range(self.Ndir):
            for k in range(self.Nk):
                k1 = self.kklst_dk[d, k]
                self.Z_dkww[d, k] = np.dot(dag(self.V_knw[k]), np.dot(
                    self.Z_dknn[d, k], self.V_knw[k1]))

        # Update the new Z matrix
        self.Z_dww = self.Z_dkww.sum(axis=1) / self.Nk

    def get_centers(self, scaled=False):
        """Calculate the Wannier centers

        ::

          pos =  L / 2pi * phase(diag(Z))
        """
        coord_wc = np.angle(self.Z_dww[:3].diagonal(0, 1, 2)).T / (2 * pi) % 1
        if not scaled:
            coord_wc = np.dot(coord_wc, self.largeunitcell_cc)
        return coord_wc

    def get_radii(self):
        r"""Calculate the spread of the Wannier functions.

        ::

                        --  /  L  \ 2       2
          radius**2 = - >   | --- |   ln |Z|
                        --d \ 2pi /
        """
        r2 = -np.dot(self.largeunitcell_cc.diagonal()**2 / (2 * pi)**2,
                     np.log(abs(self.Z_dww[:3].diagonal(0, 1, 2))**2))
        return np.sqrt(r2)

    def get_spreads(self):
        r"""Calculate the spread of the Wannier functions.
        Compute the spread as the inverse of the contribution to the spread
        functional, so that is proportional to the physical spread.

        ::

                            1
          spread_w = ---------------
                     |Z_dww|^2 * W_d


        """
        spread_dw = np.abs(self.Z_dww.diagonal(0, 1, 2))**2
        spread_w = np.dot(spread_dw.T, self.weight_d).real
        inv_spread_w = 1 / spread_w
        return inv_spread_w

    def get_spectral_weight(self, w):
        return abs(self.V_knw[:, :, w])**2 / self.Nk

    def get_pdos(self, w, energies, width):
        """Projected density of states (PDOS).

        Returns the (PDOS) for Wannier function ``w``. The calculation
        is performed over the energy grid specified in energies. The
        PDOS is produced as a sum of Gaussians centered at the points
        of the energy grid and with the specified width.
        """
        spec_kn = self.get_spectral_weight(w)
        dos = np.zeros(len(energies))
        for k, spec_n in enumerate(spec_kn):
            eig_n = self.calc.get_eigenvalues(k=k, s=self.spin)
            for weight, eig in zip(spec_n, eig_n):
                # Add gaussian centered at the eigenvalue
                x = ((energies - eig) / width)**2
                dos += weight * np.exp(-x.clip(0., 40.)) / (sqrt(pi) * width)
        return dos

    def max_spread(self, directions=[0, 1, 2]):
        """Returns the index of the most delocalized Wannier function
        together with the value of the spread functional"""
        d = np.zeros(self.nwannier)
        for dir in directions:
            d[dir] = np.abs(self.Z_dww[dir].diagonal())**2 * self.weight_d[dir]
        index = np.argsort(d)[0]
        print('Index:', index)
        print('Spread:', d[index])

    def translate(self, w, R):
        """Translate the w'th Wannier function

        The distance vector R = [n1, n2, n3], is in units of the basis
        vectors of the small cell.
        """
        for kpt_c, U_ww in zip(self.kpt_kc, self.U_kww):
            U_ww[:, w] *= np.exp(2.j * pi * np.dot(np.array(R), kpt_c))
        self.update()

    def translate_to_cell(self, w, cell):
        """Translate the w'th Wannier function to specified cell"""
        scaled_c = np.angle(self.Z_dww[:3, w, w]) * self.kptgrid / (2 * pi)
        trans = np.array(cell) - np.floor(scaled_c)
        self.translate(w, trans)

    def translate_all_to_cell(self, cell=[0, 0, 0]):
        r"""Translate all Wannier functions to specified cell.

        Move all Wannier orbitals to a specific unit cell.  There
        exists an arbitrariness in the positions of the Wannier
        orbitals relative to the unit cell. This method can move all
        orbitals to the unit cell specified by ``cell``.  For a
        `\Gamma`-point calculation, this has no effect. For a
        **k**-point calculation the periodicity of the orbitals are
        given by the large unit cell defined by repeating the original
        unitcell by the number of **k**-points in each direction.  In
        this case it is useful to move the orbitals away from the
        boundaries of the large cell before plotting them. For a bulk
        calculation with, say 10x10x10 **k** points, one could move
        the orbitals to the cell [2,2,2].  In this way the pbc
        boundary conditions will not be noticed.
        """
        scaled_wc = np.angle(self.Z_dww[:3].diagonal(0, 1, 2)).T * \
            self.kptgrid / (2 * pi)
        trans_wc = np.array(cell)[None] - np.floor(scaled_wc)
        for kpt_c, U_ww in zip(self.kpt_kc, self.U_kww):
            U_ww *= np.exp(2.j * pi * np.dot(trans_wc, kpt_c))
        self.update()

    def distances(self, R):
        Nw = self.nwannier
        cen = self.get_centers()
        r1 = cen.repeat(Nw, axis=0).reshape(Nw, Nw, 3)
        r2 = cen.copy()
        for i in range(3):
            r2 += self.unitcell_cc[i] * R[i]

        r2 = np.swapaxes(r2.repeat(Nw, axis=0).reshape(Nw, Nw, 3), 0, 1)
        return np.sqrt(np.sum((r1 - r2)**2, axis=-1))

    def get_hopping(self, R):
        """Returns the matrix H(R)_nm=<0,n|H|R,m>.

        ::

                                1   _   -ik.R
          H(R) = <0,n|H|R,m> = --- >_  e      H(k)
                                Nk  k

        where R is the cell-distance (in units of the basis vectors of
        the small cell) and n,m are indices of the Wannier functions.
        """
        H_ww = np.zeros([self.nwannier, self.nwannier], complex)
        for k, kpt_c in enumerate(self.kpt_kc):
            phase = np.exp(-2.j * pi * np.dot(np.array(R), kpt_c))
            H_ww += self.get_hamiltonian(k) * phase
        return H_ww / self.Nk

    @functools.lru_cache(maxsize=10000)
    def get_hopping_wcache(self, n1, n2, n3):
        R = np.array([n1, n2, n3], float)
        return self.get_hopping(R)

    def get_hamiltonian(self, k=0):
        """Get Hamiltonian at existing k-vector of index k

        ::

                  dag
          H(k) = V    diag(eps )  V
                  k           k    k
        """
        eps_n = self.calc.get_eigenvalues(kpt=k, spin=self.spin)[:self.nbands]
        return np.dot(dag(self.V_knw[k]) * eps_n, self.V_knw[k])

    def get_hamiltonian_kpoint(self, kpt_c):
        """Get Hamiltonian at some new arbitrary k-vector

        ::

                  _   ik.R
          H(k) = >_  e     H(R)
                  R

        Warning: This method moves all Wannier functions to cell (0, 0, 0)
        """
        if self.verbose:
            print('Translating all Wannier functions to cell (0, 0, 0)')
        self.translate_all_to_cell()
        max = (self.kptgrid - 1) // 2
        N1, N2, N3 = max
        Hk = np.zeros([self.nwannier, self.nwannier], complex)
        for n1 in range(-N1, N1 + 1):
            for n2 in range(-N2, N2 + 1):
                for n3 in range(-N3, N3 + 1):
                    R = np.array([n1, n2, n3], float)
                    hop_ww = self.get_hopping_wcache(n1, n2, n3)
                    phase = np.exp(+2.j * pi * np.dot(R, kpt_c))
                    Hk += hop_ww * phase
        return Hk

    def get_function(self, index, repeat=None):
        r"""Get Wannier function on grid.

        Returns an array with the funcion values of the indicated Wannier
        function on a grid with the size of the *repeated* unit cell.

        For a calculation using **k**-points the relevant unit cell for
        eg. visualization of the Wannier orbitals is not the original unit
        cell, but rather a larger unit cell defined by repeating the
        original unit cell by the number of **k**-points in each direction.
        Note that for a `\Gamma`-point calculation the large unit cell
        coinsides with the original unit cell.
        The large unitcell also defines the periodicity of the Wannier
        orbitals.

        ``index`` can be either a single WF or a coordinate vector in terms
        of the WFs.
        """

        # Default size of plotting cell is the one corresponding to k-points.
        if repeat is None:
            repeat = self.kptgrid
        N1, N2, N3 = repeat

        dim = self.calc.get_number_of_grid_points()
        largedim = dim * [N1, N2, N3]

        wanniergrid = np.zeros(largedim, dtype=complex)
        for k, kpt_c in enumerate(self.kpt_kc):
            # The coordinate vector of wannier functions
            if isinstance(index, int):
                vec_n = self.V_knw[k, :, index]
            else:
                vec_n = np.dot(self.V_knw[k], index)

            wan_G = np.zeros(dim, complex)
            for n, coeff in enumerate(vec_n):
                wan_G += coeff * self.calc.get_pseudo_wave_function(
                    n, k, self.spin, pad=True)

            # Distribute the small wavefunction over large cell:
            for n1 in range(N1):
                for n2 in range(N2):
                    for n3 in range(N3):  # sign?
                        e = np.exp(-2.j * pi * np.dot([n1, n2, n3], kpt_c))
                        wanniergrid[n1 * dim[0]:(n1 + 1) * dim[0],
                                    n2 * dim[1]:(n2 + 1) * dim[1],
                                    n3 * dim[2]:(n3 + 1) * dim[2]] += e * wan_G

        # Normalization
        wanniergrid /= np.sqrt(self.Nk)
        return wanniergrid

    def write_cube(self, index, fname, repeat=None, real=True):
        """Dump specified Wannier function to a cube file"""
        from ase.io import write

        # Default size of plotting cell is the one corresponding to k-points.
        if repeat is None:
            repeat = self.kptgrid
        atoms = self.calc.get_atoms() * repeat
        func = self.get_function(index, repeat)

        # Handle separation of complex wave into real parts
        if real:
            if self.Nk == 1:
                func *= np.exp(-1.j * np.angle(func.max()))
                if 0:
                    assert max(abs(func.imag).flat) < 1e-4
                func = func.real
            else:
                func = abs(func)
        else:
            phase_fname = fname.split('.cube')
            phase_fname.insert(1, 'phase')
            phase_fname = '.'.join(phase_fname) + 'cube'
            write(phase_fname, atoms, data=np.angle(func), format='cube')
            func = abs(func)

        write(fname, atoms, data=func, format='cube')

    # def localize(self, step=0.005, tolerance=1e-08,
    def localize(self, step=0.25, tolerance=1e-08,
                 updaterot=True, updatecoeff=True):
        """Optimize rotation to give maximal localization"""
        # steepest_descent(self, step, tolerance, verbose=self.verbose,
        md_min(self, step=step, tolerance=tolerance, verbose=self.verbose,
               updaterot=updaterot, updatecoeff=updatecoeff)

    def get_functional_value(self):
        """Calculate the value of the spread functional.

        ::

          Tr[|ZI|^2]=sum(I)sum(n) w_i|Z_(i)_nn|^2,

        where w_i are weights."""
        if self.functional == 'std':
            a_d = np.sum(np.abs(self.Z_dww.diagonal(0, 1, 2))**2, axis=1)
            fun = np.dot(a_d, self.weight_d).real
        elif self.functional == 'sigmoid':
            a_d = np.sum(1 / (1 + np.exp(-10 * (np.abs(
                self.Z_dww.diagonal(0, 1, 2))**2 - 0.5))),
                axis=1)
            fun = np.dot(a_d, self.weight_d).real
        elif self.functional == 'erf':
            a_d = np.sum(erf(2 * np.abs(self.Z_dww.diagonal(0, 1, 2))**2),
                         axis=1)
            fun = np.dot(a_d, self.weight_d).real
        elif self.functional == 'sqrt':
            a_d = np.sum(np.abs(self.Z_dww.diagonal(0, 1, 2)), axis=1)
            fun = np.dot(a_d, self.weight_d).real
        elif self.functional == 'cbrt':
            a_d = np.sum(np.cbrt(np.abs(self.Z_dww.diagonal(0, 1, 2))**2),
                         axis=1)
            fun = np.dot(a_d, self.weight_d).real
        elif self.functional == 'stdev':
            a_dw = np.abs(self.Z_dww.diagonal(0, 1, 2))**2
            a_w = np.dot(a_dw.T, self.weight_d).real
            fun = np.sum(a_w) - 10 * np.var(a_w)
            if self.verbose:
                print(f'std: {np.sum(a_w):.4f} \tvar: {10 * np.var(a_w):.4f}')
        return fun

    def get_gradients(self):
        # Determine gradient of the spread functional.
        #
        # The gradient for a rotation A_kij is::
        #
        #    dU = dRho/dA_{k,i,j} = sum(I) sum(k')
        #            + Z_jj Z_kk',ij^* - Z_ii Z_k'k,ij^*
        #            - Z_ii^* Z_kk',ji + Z_jj^* Z_k'k,ji
        #
        # The gradient for a change of coefficients is::
        #
        #   dRho/da^*_{k,i,j} = sum(I) [[(Z_0)_{k} V_{k'} diag(Z^*) +
        #                                (Z_0_{k''})^d V_{k''} diag(Z)] *
        #                                U_k^d]_{N+i,N+j}
        #
        # where diag(Z) is a square,diagonal matrix with Z_nn in the diagonal,
        # k' = k + dk and k = k'' + dk.
        #
        # The extra degrees of freedom chould be kept orthonormal to the fixed
        # space, thus we introduce lagrange multipliers, and minimize instead::
        #
        #     Rho_L = Rho - sum_{k,n,m} lambda_{k,nm} <c_{kn}|c_{km}>
        #
        # for this reason the coefficient gradients should be multiplied
        # by (1 - c c^dag).

        Nb = self.nbands
        Nw = self.nwannier

        if self.functional == 'stdev':
            O_dw = np.abs(self.Z_dww.diagonal(0, 1, 2))**2
            O_w = np.dot(O_dw.T, self.weight_d).real
            O = np.sum(O_w)

        dU = []
        dC = []
        for k in range(self.Nk):
            M = self.fixedstates_k[k]
            L = self.edf_k[k]
            U_ww = self.U_kww[k]
            C_ul = self.C_kul[k]
            Utemp_ww = np.zeros((Nw, Nw), complex)
            Ctemp_nw = np.zeros((Nb, Nw), complex)

            for d, weight in enumerate(self.weight_d):
                if abs(weight) < 1.0e-6:
                    continue

                Z_knn = self.Z_dknn[d]
                diagZ_w = self.Z_dww[d].diagonal()
                Zii_ww = np.repeat(diagZ_w, Nw).reshape(Nw, Nw)
                if self.functional == 'sigmoid':
                    diagZ_w = (
                        (1 / (1 + np.exp(-10
                                         * (diagZ_w * diagZ_w.conj()
                                            - 0.5))))
                        * (1 - (1 / (1 + np.exp(-10
                                                * (diagZ_w * diagZ_w.conj()
                                                   - 0.5)))))
                        * diagZ_w)
                    Zii_ww = (
                        (1 / (1 + np.exp(-10
                                         * (Zii_ww * Zii_ww.conj() - 0.5))))
                        * (1 - (1 / (1 + np.exp(-10
                                                * (Zii_ww * Zii_ww.conj()
                                                   - 0.5)))))
                        * Zii_ww)
                elif self.functional == 'erf':
                    diagZ_w = ((4 / np.sqrt(np.pi))
                               * np.exp(-(2 * diagZ_w * diagZ_w.conj())**2)
                               * diagZ_w)
                    Zii_ww = ((4 / np.sqrt(np.pi))
                              * np.exp(-(2 * Zii_ww * Zii_ww.conj())**2)
                              * Zii_ww)
                elif self.functional == 'sqrt':
                    diagZ_w = 0.5 * (1 / np.sqrt(diagZ_w * diagZ_w.conj())
                                     * diagZ_w)
                    Zii_ww = 0.5 * (1 / np.sqrt(Zii_ww * Zii_ww.conj())
                                    * Zii_ww)
                elif self.functional == 'cbrt':
                    diagZ_w = ((1 / 3)
                               * 1 / np.power(diagZ_w * diagZ_w.conj(),
                                              2 / 3)
                               * diagZ_w)
                    Zii_ww = ((1 / 3)
                              * 1 / np.power(Zii_ww * Zii_ww.conj(), 2 / 3)
                              * Zii_ww)
                elif self.functional == 'stdev':
                    diagOZ_w = O_w * diagZ_w
                    OZii_ww = np.repeat(diagOZ_w, Nw).reshape(Nw, Nw)

                k1 = self.kklst_dk[d, k]
                k2 = self.invkklst_dk[d, k]
                V_knw = self.V_knw
                Z_kww = self.Z_dkww[d]

                if L > 0:
                    Ctemp_nw += weight * np.dot(
                        np.dot(Z_knn[k], V_knw[k1]) * diagZ_w.conj() +
                        np.dot(dag(Z_knn[k2]), V_knw[k2]) * diagZ_w,
                        dag(U_ww))

                    if self.functional == 'stdev':
                        Ctemp_nw += 20 * O * weight * np.dot(
                            np.dot(Z_knn[k], V_knw[k1]) * diagZ_w.conj() +
                            np.dot(dag(Z_knn[k2]), V_knw[k2]) * diagZ_w,
                            dag(U_ww)) / Nw**2

                        Ctemp_nw -= 20 * weight * np.dot(
                            np.dot(Z_knn[k], V_knw[k1]) * diagOZ_w.conj() +
                            np.dot(dag(Z_knn[k2]), V_knw[k2]) * diagOZ_w,
                            dag(U_ww)) / Nw

                temp = Zii_ww.T * Z_kww[k].conj() - Zii_ww * Z_kww[k2].conj()
                Utemp_ww += weight * (temp - dag(temp))

                if self.functional == 'stdev':
                    Utemp_ww += 20 * O * weight * (temp - dag(temp)) / Nw**2

                    temp = (OZii_ww.T * Z_kww[k].conj()
                            - OZii_ww * Z_kww[k2].conj())
                    Utemp_ww -= 20 * weight * (temp - dag(temp)) / Nw

            dU.append(Utemp_ww.ravel())

            if L > 0:
                # Ctemp now has same dimension as V, the gradient is in the
                # lower-right (Nb-M) x L block
                Ctemp_ul = Ctemp_nw[M:, M:]
                G_ul = Ctemp_ul - np.dot(np.dot(C_ul, dag(C_ul)), Ctemp_ul)
                dC.append(G_ul.ravel())

        return np.concatenate(dU + dC)

    def step(self, dX, updaterot=True, updatecoeff=True):
        # dX is (A, dC) where U->Uexp(-A) and C->C+dC
        Nw = self.nwannier
        Nk = self.Nk
        M_k = self.fixedstates_k
        L_k = self.edf_k
        if updaterot:
            A_kww = dX[:Nk * Nw**2].reshape(Nk, Nw, Nw)
            for U, A in zip(self.U_kww, A_kww):
                H = -1.j * A.conj()
                epsilon, Z = np.linalg.eigh(H)
                # Z contains the eigenvectors as COLUMNS.
                # Since H = iA, dU = exp(-A) = exp(iH) = ZDZ^d
                dU = np.dot(Z * np.exp(1.j * epsilon), dag(Z))
                if U.dtype == float:
                    U[:] = np.dot(U, dU).real
                else:
                    U[:] = np.dot(U, dU)

        if updatecoeff:
            start = 0
            for C, unocc, L in zip(self.C_kul, self.nbands - M_k, L_k):
                if L == 0 or unocc == 0:
                    continue
                Ncoeff = L * unocc
                deltaC = dX[Nk * Nw**2 + start: Nk * Nw**2 + start + Ncoeff]
                C += deltaC.reshape(unocc, L)
                gram_schmidt(C)
                start += Ncoeff

        self.update()
