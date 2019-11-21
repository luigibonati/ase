"""Quantum ESPRESSO Calculator

Remember to:

export ESPRESSO_HOME="/path/to/espresso/bin" (path that contains your pw.x)
export ESPRESSO_PSEUDO="path/to/espresso/pseudo"
"""

import os
import warnings
import subprocess
from ase import io
from ase.calculators.calculator import FileIOCalculator, PropertyNotPresent


error_template = 'Property "%s" not available. Please try running Quantum\n' \
                 'Espresso first by calling Atoms.get_potential_energy().'

warn_template = 'Property "%s" is None. Typically, this is because the ' \
                'required information has not been printed by Quantum ' \
                'Espresso at a "low" verbosity level (the default). ' \
                'Please try running Quantum Espresso with "high" verbosity.'


class EspressoProfile():
    # Initial Espresso calculator profile based on Ask's template
    # Not sure where this will go eventually

    def __init__(self):
        self.name = 'espresso'
        # Location of bin
        self.espresso_home = os.environ.get('ESPRESSO_HOME', '')

        if os.path.isfile(self.espresso_home + '/pw.x'):
            self.command = \
                self.espresso_home + '/pw.x -in PREFIX.pwi > PREFIX.pwo'
        else:
            self.command = None

        self.pseudo_path = None
        home = os.environ.get('HOME', '')
        pseudo_path = os.environ.get('ESPRESSO_PSEUDO')
        # Looking for pseudopotential in different places
        if pseudo_path:
            if os.path.isdir(pseudo_path):
                self.pseudo_path = pseudo_path
        elif os.path.isdir(home + '/espresso/pseudo/'):
            self.pseudo_path = home + '/espresso/pseudo/'
        elif os.path.isdir(self.espresso_home.replace('/bin', '/pseudo')):
            self.pseudo_path = self.espresso_home.replace('/bin', '/pseudo')

        self.version = self.get_version()

    def get_version(self):
        if not self.command:
            return None
        plain_command = self.espresso_home + '/pw.x'

        proc = subprocess.Popen([plain_command], stdout=subprocess.PIPE)
        # This is a little messy - running QE to get the version
        # from the output. It there a better way to do this?
        try:
            outs, errs = proc.communicate(timeout=1)
        except subprocess.TimeoutExpired:
            proc.kill()
            outs, errs = proc.communicate()

        lines = outs.splitlines()
        version = None
        for l in lines:
            l = str(l)
            if 'Program PWSCF' in str(l):
                version = l.split('PWSCF ')[-1].split(' starts')[0]
        return version

    def available(self):
        return bool(self.version) and bool(self.pseudo_path)


class Espresso(FileIOCalculator, EspressoProfile):
    """
    """
    implemented_properties = ['energy', 'forces', 'stress', 'magmoms']

    def __init__(self, restart=None, ignore_bad_restart_file=False,
                 label='espresso', atoms=None, **kwargs):
        """
        All options for pw.x are copied verbatim to the input file, and put
        into the correct section. Use ``input_data`` for parameters that are
        already in a dict, all other ``kwargs`` are passed as parameters.

        Accepts all the options for pw.x as given in the QE docs, plus some
        additional options:

        input_data: dict
            A flat or nested dictionary with input parameters for pw.x
        pseudopotentials: dict
            A filename for each atomic species, e.g.
            ``{'O': 'O.pbe-rrkjus.UPF', 'H': 'H.pbe-rrkjus.UPF'}``.
            A dummy name will be used if none are given.
        kspacing: float
            Generate a grid of k-points with this as the minimum distance,
            in A^-1 between them in reciprocal space. If set to None, kpts
            will be used instead.
        kpts: (int, int, int), dict, or BandPath
            If kpts is a tuple (or list) of 3 integers, it is interpreted
            as the dimensions of a Monkhorst-Pack grid.
            If kpts is a dict, it will either be interpreted as a path
            in the Brillouin zone (*) if it contains the 'path' keyword,
            otherwise it is converted to a Monkhorst-Pack grid (**).
            (*) see ase.dft.kpoints.bandpath
            (**) see ase.calculators.calculator.kpts2sizeandoffsets
        koffset: (int, int, int)
            Offset of kpoints in each direction. Must be 0 (no offset) or
            1 (half grid offset). Setting to True is equivalent to (1, 1, 1).


        .. note::
           Set ``tprnfor=True`` and ``tstress=True`` to calculate forces and
           stresses.

        .. note::
           Band structure plots can be made as follows:


           1. Perform a regular self-consistent calculation,
              saving the wave functions at the end, as well as
              getting the Fermi energy:

              >>> input_data = {<your input data>}
              >>> calc = Espresso(input_data=input_data, ...)
              >>> atoms.set_calculator(calc)
              >>> atoms.get_potential_energy()
              >>> fermi_level = calc.get_fermi_level()

           2. Perform a non-self-consistent 'band structure' run
              after updating your input_data and kpts keywords:

              >>> input_data['control'].update({'calculation':'bands',
              >>>                               'restart_mode':'restart',
              >>>                               'verbosity':'high'})
              >>> calc.set(kpts={<your Brillouin zone path>},
              >>>          input_data=input_data)
              >>> calc.calculate(atoms)

           3. Make the plot using the BandStructure functionality,
              after setting the Fermi level to that of the prior
              self-consistent calculation:

              >>> bs = calc.band_structure()
              >>> bs.reference = fermi_energy
              >>> bs.plot()

        """

        EspressoProfile.__init__(self)
        FileIOCalculator.__init__(self, restart, ignore_bad_restart_file,
                                  label, atoms, **kwargs)

        self.calc = None

    def set(self, **kwargs):
        changed_parameters = FileIOCalculator.set(self, **kwargs)
        if changed_parameters:
            self.reset()

    def write_input(self, atoms, properties=None, system_changes=None):
        FileIOCalculator.write_input(self, atoms, properties, system_changes)
        io.write(self.label + '.pwi', atoms, **self.parameters)

    def read_results(self):
        output = io.read(self.label + '.pwo')
        self.calc = output.calc
        self.results = output.calc.results

    def get_fermi_level(self):
        if self.calc is None:
            raise PropertyNotPresent(error_template % 'Fermi level')
        return self.calc.get_fermi_level()

    def get_ibz_k_points(self):
        if self.calc is None:
            raise PropertyNotPresent(error_template % 'IBZ k-points')
        ibzkpts = self.calc.get_ibz_k_points()
        if ibzkpts is None:
            warnings.warn(warn_template % 'IBZ k-points')
        return ibzkpts

    def get_k_point_weights(self):
        if self.calc is None:
            raise PropertyNotPresent(error_template % 'K-point weights')
        k_point_weights = self.calc.get_k_point_weights()
        if k_point_weights is None:
            warnings.warn(warn_template % 'K-point weights')
        return k_point_weights

    def get_eigenvalues(self, **kwargs):
        if self.calc is None:
            raise PropertyNotPresent(error_template % 'Eigenvalues')
        eigenvalues = self.calc.get_eigenvalues(**kwargs)
        if eigenvalues is None:
            warnings.warn(warn_template % 'Eigenvalues')
        return eigenvalues

    def get_number_of_spins(self):
        if self.calc is None:
            raise PropertyNotPresent(error_template % 'Number of spins')
        nspins = self.calc.get_number_of_spins()
        if nspins is None:
            warnings.warn(warn_template % 'Number of spins')
        return nspins
