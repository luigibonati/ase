"""This module defines an ASE interface to ABINIT.

http://www.abinit.org/
"""

import re

import ase.io.abinit as io
from ase.calculators.calculator import FileIOCalculator
from ase.calculators.genericfileio import (CalculatorTemplate,
                                           GenericFileIOCalculator)
from subprocess import check_output
from pathlib import Path


def get_abinit_version(command):
    txt = check_output([command, '--version']).decode('ascii')
    # This allows trailing stuff like betas, rc and so
    m = re.match(r'\s*(\d\.\d\.\d)', txt)
    if m is None:
        raise RuntimeError('Cannot recognize abinit version. '
                           'Start of output: {}'
                           .format(txt[:40]))
    return m.group(1)


class AbinitProfile:
    def __init__(self, argv):
        self.argv = argv

    def version(self):
        from subprocess import check_output
        return check_output(self.argv + ['--version'])

    def run(self, directory, inputfile, outputfile):
        from subprocess import check_call
        with open(outputfile, 'w') as fd:
            check_call(self.argv + [str(inputfile)], stdout=fd,
                       cwd=directory)


class AbinitTemplate(CalculatorTemplate):
    _label = 'abinit'  # Controls naming of files within calculation directory

    def __init__(self):
        self.name = 'abinit'  # XXX
        self.implemented_properties = Abinit.implemented_properties
        self.input_file = 'abinit.in'
        self.output_file = 'abinit.xxxxxxx'
        # XXXXXXXXX constructor or something?

    def write_input(self, directory, atoms, parameters, properties):
        directory = Path(directory)
        directory.mkdir(exist_ok=True, parents=True)
        parameters = dict(parameters)
        pp_paths = parameters.pop('pp_paths', None)
        assert pp_paths is not None

        kw = dict(
            xc='LDA',
            smearing=None,
            kpts=None,
            raw=None,
            pps='fhi')
        kw.update(parameters)

        io.prepare_abinit_input(
            directory=directory,
            atoms=atoms, properties=properties, parameters=kw,
            pp_paths=pp_paths)

    def read_results(self, directory):
        return io.read_abinit_outputs(directory)



class Abinit(GenericFileIOCalculator):
    """Class for doing ABINIT calculations.

    The default parameters are very close to those that the ABINIT
    Fortran code would use.  These are the exceptions::

      calc = Abinit(label='abinit', xc='LDA', ecut=400, toldfe=1e-5)
    """

    implemented_properties = ['energy', 'forces', 'stress', 'magmom']
    #ignored_changes = {'pbc'}  # In abinit, pbc is always effectively True.
    #command = 'abinit < PREFIX.files > PREFIX.log'
    #discard_results_on_any_change = True

    #default_parameters = dict(
    #    xc='LDA',
    #    smearing=None,
    #    kpts=None,
    #    raw=None,
    #    pps='fhi')

    def __init__(self, *, # restart=None,
                 profile=None,
                 directory='.',
                 # ignore_bad_restart_file=FileIOCalculator._deprecated,
                 # label='abinit', atoms=None, pp_paths=None,
                 **kwargs):
        """Construct ABINIT-calculator object.

        Parameters
        ==========
        label: str
            Prefix to use for filenames (label.in, label.txt, ...).
            Default is 'abinit'.

        Examples
        ========
        Use default values:

        >>> h = Atoms('H', calculator=Abinit(ecut=200, toldfe=0.001))
        >>> h.center(vacuum=3.0)
        >>> e = h.get_potential_energy()

        """

        if profile is None:
            profile = AbinitProfile(['abinit'])

        assert 'pp_paths' in kwargs

        super().__init__(template=AbinitTemplate(),
                         profile=profile,
                         directory=directory,
                         parameters=kwargs)

        # FileIOCalculator.__init__(self, restart, ignore_bad_restart_file,
        #                           label, atoms, **kwargs)

    #def write_input(self, atoms, properties, system_changes):
    #    """Write input parameters to files-file."""

    #    io.prepare_abinit_input(
    #        directory=self.directory,
    #        atoms=atoms, properties=properties, parameters=self.parameters,
    #        pp_paths=self.pp_paths)

    #def read_results(self):
    #    self.results = io.read_abinit_outputs(self.directory)

    #def get_number_of_iterations(self):
    #    return self.results['niter']

    #def get_electronic_temperature(self):
    #    return self.results['width']

    #def get_number_of_electrons(self):
    #    return self.results['nelect']

    #def get_number_of_bands(self):
    #    return self.results['nbands']

    #def get_k_point_weights(self):
    #    return self.results['kpoint_weights']

    #def get_bz_k_points(self):
    #    raise NotImplementedError

    #def get_ibz_k_points(self):
    #    return self.results['ibz_kpoints']

    #def get_spin_polarized(self):
    #    return self.results['eigenvalues'].shape[0] == 2

    #def get_number_of_spins(self):
    #    return len(self.results['eigenvalues'])

    #def get_fermi_level(self):
    #    return self.results['fermilevel']

    #def get_eigenvalues(self, kpt=0, spin=0):
    #    return self.results['eigenvalues'][spin, kpt]

    #def get_occupations(self, kpt=0, spin=0):
    #    raise NotImplementedError
