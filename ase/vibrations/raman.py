import numpy as np

from ase.vibrations import Vibrations
from ase.utils.timing import Timer
from ase.utils import convert_string_to_fd


class Raman(Vibrations):
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


class RamanCalculator(Raman):
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
    

class RamanData(Raman):
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
