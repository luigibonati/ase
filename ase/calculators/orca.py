import re
import ase.io.orca as io
from ase.calculators.genericfileio import (CalculatorTemplate,
                                           GenericFileIOCalculator)
from pathlib import Path


def get_version_from_orca_header(orca_header):
    match = re.search(r'Program Version (\S+)', orca_header, re.M)
    return match.group(1)


class OrcaProfile:
    def __init__(self, argv):
        self.argv = argv

    def version(self):
        # XXX Allow MPI in argv; the version call should not be parallel.
        from ase.calculators.genericfileio import read_stdout
        stdout = read_stdout([*self.argv, "does_not_exist"])
        return get_version_from_orca_header(stdout)

    def run(self, directory, inputfile, outputfile):
        from subprocess import check_call
        with open(outputfile, 'w') as fd:
            check_call(self.argv + [str(inputfile)], stdout=fd, cwd=directory)


class OrcaTemplate(CalculatorTemplate):
    label = 'orca'

    def __init__(self):
        super().__init__(name='orca',
                         implemented_properties=['energy', 'forces'])

        self.input_file = f'{self.label}.inp'
        self.output_file = f'{self.label}.out'

    def execute(self, directory, profile) -> None:
        profile.run(directory, self.input_file, self.output_file)

    def write_input(self, directory, atoms, parameters, properties):
        directory = Path(directory)
        directory.mkdir(exist_ok=True, parents=True)
        parameters = dict(parameters)

        kw = dict(charge=0, mult=1, orcasimpleinput='B3LYP def2-TZVP',
                  orcablocks='%pal nprocs 1 end')
        kw.update(parameters)

        io.write_orca(atoms=atoms, **kw)

    def read_results(self, directory):
        return io.read_orca_outputs(directory, self.label)


class ORCA(GenericFileIOCalculator):
    """Class for doing Orca calculations.

    The default parameters are very close to those that the ABINIT
    Fortran code would use.  These are the exceptions::

      calc = Abinit(label='orca', charge=0, mult=1, orcasimpleinput='B3LYP def2-TZVP',
        orcablocks='%pal nprocs 16 end')
    """

    def __init__(self, *, profile=None, directory='.', **kwargs):
        """Construct ORCA-calculator object.

        Parameters
        ==========
        label: str
            Prefix to use for filenames (label.inp, label.out).
            Default is 'orca'.
        charge: int

        mult: int

        orcasimpleinput : str

        orcablocks: str


        Examples
        ========
        Use default values:


        >>> h = Atoms('H', calculator=Orca(charge=0,mult=1,label='water',
        orcasimpleinput='B3LYP def2-TZVP',
        orcablocks='%pal nprocs 16 end')

        """

        if profile is None:
            profile = OrcaProfile(['orca'])

        super().__init__(template=OrcaTemplate(),
                         profile=profile, directory=directory,
                         parameters=kwargs)
