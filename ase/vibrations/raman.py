import numpy as np

from ase.vibrations import Vibrations
from ase.utils.timing import Timer
from ase.utils import convert_string_to_fd
import ase.units as u


class RamanBase(Vibrations):
    def __init__(self, atoms,  # XXX do we need atoms at this stage ?
                 gsname='rraman',  # name for ground state calculations
                 exname=None,      # name for excited state calculations
                 exext='.alpha',   # extension for Excitation names
                 txt='-',
                 *args, **kwargs):
        """
        Parameters
        ----------
        atoms: ase Atoms object
        gsname: string
            name for ground state calculations, used in run() and
            for reading of forces (default 'rraman')
        exname: string
            name for excited state calculations (defaults to gsname),
            used for reading excitations
        exext: string
            Extension for excitation filenames
        txt:
            Output stream
        """
        Vibrations.__init__(self, atoms, *args, **kwargs)

        self.name = gsname
        if exname is None:
            exname = gsname
        self.exname = exname
        self.exext = exext
        
        self.timer = Timer()
        self.txt = convert_string_to_fd(txt)


class RamanCalculator(RamanBase):
    """Base class for Raman calculators"""
    def __init__(self, atoms, Calculator, *args, **kwargs):
        self.exobj = Calculator
        Raman.__init__(self, atoms, *args, **kwargs)


class RamanStaticCalculator(RamanCalculator):
    """Base class for Raman intensities derived from
    static polarizabilities"""
    def calculate(self, atoms, filename, fd):
        # write forces
        Vibrations.calculate(self, atoms, filename, fd)
        # write static polarizability
        fname = filename.replace('.pckl', self.exext)
        np.savetxt(fname, self.exobj().calculate(atoms))
    

class Raman(RamanBase):
    """Base class to evaluate Raman spectra from pre-computed data"""
    def init_parallel_read(self):
        """Initialize variables for parallel read"""
        rank = self.comm.rank
        self.ndof = 3 * len(self.indices)
        myn = -(-self.ndof // self.comm.size)  # ceil divide
        self.slize = s = slice(myn * rank, myn * (rank + 1))
        self.myindices = np.repeat(self.indices, 3)[s]
        self.myxyz = ('xyz' * len(self.indices))[s]
        self.myr = range(self.ndof)[s]
        self.mynd = len(self.myr)

    def read(self, method='standard', direction='central'):
        """Read data from a pre-performed calculation."""

        self.timer.start('read')
        self.timer.start('vibrations')
        Vibrations.read(self, method, direction)
        # we now have:
        # self.H     : Hessian matrix
        # self.im    : 1./sqrt(masses)
        # self.modes : Eigenmodes of the mass weighted Hessian
        self.om_Q = self.hnu.real    # energies in eV
        self.om_v = self.om_Q
        # pre-factors for one vibrational excitation
        with np.errstate(divide='ignore'):
            self.vib01_Q = np.where(self.om_Q > 0,
                                    1. / np.sqrt(2 * self.om_Q), 0)
        # -> sqrt(amu) * Angstrom
        self.vib01_Q *= np.sqrt(u.Ha * u._me / u._amu) * u.Bohr
        self.timer.stop('vibrations')

        self.timer.start('excitations')
        self.init_parallel_read()
        if not hasattr(self, 'ex0E_p'):
            if self.overlap:
                self.read_excitations_overlap()
            else:
                self.read_excitations()
        self.timer.stop('excitations')
        self.timer.stop('read')

    @staticmethod
    def m2(z):
        return (z * z.conj()).real

    def me_Qcc(self, *args, **kwargs):
        """Full matrix element

        Returns
        -------
        Matrix element in e^2 Angstrom^2 / eV
        """
        # Angstrom^2 / sqrt(amu)
        elme_Qcc = self.electronic_me_Qcc(*args, **kwargs)
        # Angstrom^3 -> e^2 Angstrom^2 / eV
        elme_Qcc /= u.Hartree * u.Bohr  # e^2 Angstrom / eV / sqrt(amu)
        return elme_Qcc * self.vib01_Q[:, None, None]

    def get_cross_sections(self, *args, **kwargs):
        """Returns Raman cross sections for each vibration."""
        I_v = self.intensity(*args, **kwargs)
        pre = 1. / 16 / np.pi**2 / u._eps0**2 / u._c**4
        # frequency of scattered light
        omS_v = omega - self.om_v
        return pre * omega * omS_v**3 * I_v

    def absolute_intensity(self, *args, delta=0, **kwargs):
        """Absolute Raman intensity or Raman scattering factor

        Parameter
        ---------
        omega: float
           incoming laser energy, unit eV
        gamma: float
           width (imaginary energy), unit eV
        delta: float
           pre-factor for asymmetric anisotropy, default 0

        References
        ----------
        Porezag and Pederson, PRB 54 (1996) 7830-7836 (delta=0)
        Baiardi and Barone, JCTC 11 (2015) 3267-3280 (delta=5)

        Returns
        -------
        raman intensity, unit Ang**4/amu
        """

        alpha2_r, gamma2_r, delta2_r = self._invariants(
            self.electronic_me_Qcc(*args, **kwargs))
        return 45 * alpha2_r + delta * delta2_r + 7 * gamma2_r

    def intensity(self, *args, **kwargs):
        """Raman intensity

        Returns
        -------
        unit e^4 Angstrom^4 / eV^2
        """
        m2 = ResonantRaman.m2
        alpha_Qcc = self.me_Qcc(*args, **kwargs)
        if not self.observation:  # XXXX remove
            """Simple sum, maybe too simple"""
            return m2(alpha_Qcc).sum(axis=1).sum(axis=1)
        # XXX enable when appropriate
        #        if self.observation['orientation'].lower() != 'random':
        #            raise NotImplementedError('not yet')

        # random orientation of the molecular frame
        # Woodward & Long,
        # Guthmuller, J. J. Chem. Phys. 2016, 144 (6), 64106
        alpha2_r, gamma2_r, delta2_r = self._invariants(alpha_Qcc)

        if self.observation['geometry'] == '-Z(XX)Z':  # Porto's notation
            return (45 * alpha2_r + 5 * delta2_r + 4 * gamma2_r) / 45.
        elif self.observation['geometry'] == '-Z(XY)Z':  # Porto's notation
            return gamma2_r / 15.
        elif self.observation['scattered'] == 'Z':
            # scattered light in direction of incoming light
            return (45 * alpha2_r + 5 * delta2_r + 7 * gamma2_r) / 45.
        elif self.observation['scattered'] == 'parallel':
            # scattered light perendicular and
            # polarization in plane
            return 6 * gamma2_r / 45.
        elif self.observation['scattered'] == 'perpendicular':
            # scattered light perendicular and
            # polarization out of plane
            return (45 * alpha2_r + 5 * delta2_r + 7 * gamma2_r) / 45.
        else:
            raise NotImplementedError

    def _invariants(self, alpha_Qcc):
        """Raman invariants

        Parameter
        ---------
        alpha_Qcc: array
           Matrix element or polarizability tensor

        Reference
        ---------
        Derek A. Long, The Raman Effect, ISBN 0-471-49028-8

        Returns
        -------
        mean polarizability, anisotropy, asymmetric anisotropy
        """
        m2 = Raman.m2
        alpha2_r = m2(alpha_Qcc[:, 0, 0] + alpha_Qcc[:, 1, 1] +
                      alpha_Qcc[:, 2, 2]) / 9.
        delta2_r = 3 / 4. * (
            m2(alpha_Qcc[:, 0, 1] - alpha_Qcc[:, 1, 0]) +
            m2(alpha_Qcc[:, 0, 2] - alpha_Qcc[:, 2, 0]) +
            m2(alpha_Qcc[:, 1, 2] - alpha_Qcc[:, 2, 1]))
        gamma2_r = (3 / 4. * (m2(alpha_Qcc[:, 0, 1] + alpha_Qcc[:, 1, 0]) +
                              m2(alpha_Qcc[:, 0, 2] + alpha_Qcc[:, 2, 0]) +
                              m2(alpha_Qcc[:, 1, 2] + alpha_Qcc[:, 2, 1])) +
                    (m2(alpha_Qcc[:, 0, 0] - alpha_Qcc[:, 1, 1]) +
                     m2(alpha_Qcc[:, 0, 0] - alpha_Qcc[:, 2, 2]) +
                     m2(alpha_Qcc[:, 1, 1] - alpha_Qcc[:, 2, 2])) / 2)
        return alpha2_r, gamma2_r, delta2_r
