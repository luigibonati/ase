import numpy as np

from ase.vibrations import Vibrations
from ase.utils.timing import Timer
from ase.utils import convert_string_to_fd


class RamanCalculator(Vibrations):
    """Base class for Raman calculators"""
    def __init__(self, atoms, Calculator,
                 gsname='rraman',  # name for ground state calculations
                 exname=None,      # name for excited state calculations
                 txt='-',
                 *args, **kwargs):
        Vibrations.__init__(self, atoms, *args, **kwargs)

        self.exobj = Calculator
        self.name = gsname
        if exname is None:
            exname = gsname
        self.exname = exname
        # self.exext = exext  # XXX what to do?
        
        self.timer = Timer()
        self.txt = convert_string_to_fd(txt)


class RamanStaticCalculator(RamanCalculator):
    """Base class for Raman intensities derived from 
    static polarizabilities"""
    def calculate(self, atoms, filename, fd):
        # write forces
        Vibrations.calculate(self, atoms, filename, fd)
        # write static polarizability
        fname = filename.replace('.pckl', '.alpha')
        np.savetxt(fname, self.exobj().calculate(atoms))


class RamanStaticData(Vibrations):
    pass
