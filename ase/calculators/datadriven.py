import sys
import warnings
from ase.io import read, write
from ase.io.formats import ioformats
from ase.calculators.calculator import FileIOCalculator, PropertyNotPresent


class CalculatorTemplate:
    def __init__(self, name, implemented_properties, command,
                 input_file, output_file,
                 input_format, output_format):
        self.name = name
        self.implemented_properties = implemented_properties
        self.command = command

        # Generalize: We need some kind of Writer and Reader
        # to handle multiple files at a time.
        self.input_file = input_file
        self.output_file = output_file
        self.input_format = input_format
        self.output_format = output_format

    def __repr__(self):
        return 'CalculatorTemplate({})'.format(vars(self))

    def new(self, **kwargs):
        calc = DataDrivenCalculator(template=self, **kwargs)
        return calc


def get_espresso_template():
    from ase.calculators.espresso import Espresso
    infile = 'espresso.pwi'
    outfile = 'espresso.pwo'
    return CalculatorTemplate(
        name='espresso',
        implemented_properties=Espresso.implemented_properties,
        command='pw.x -in {} > {}'.format(infile, outfile),
        input_file=infile,
        output_file=outfile,
        input_format='espresso-in',
        output_format='espresso-out')


def get_emt_template():
    from ase.calculators.emt import EMT
    infile = 'input.traj'
    outfile = 'output.traj'
    return CalculatorTemplate(
        name='emt',
        implemented_properties=EMT.implemented_properties,
        command=('{} -m ase.calculators.emt {} {}'
                 .format(sys.executable, infile, outfile)),
        input_file=infile,
        output_file=outfile,
        input_format='traj',
        output_format='traj')

def get_openmx_template():
    runfile = 'openmx.dat'
    outfile = 'openmx.log'
    return CalculatorTemplate(
        name='openmx',
        implemented_properties=['energy', 'free_energy', 'forces'],
        command='openmx {} > {}'.format(runfile, outfile),
        input_file=runfile,
        output_file=outfile,
        input_format='openmx-in',
        output_format='openmx-out')


def new_espresso(**kwargs):
    return get_espresso_template().new(**kwargs)

def new_emt(**kwargs):
    return get_emt_template().new(**kwargs)

def new_openmx(**kwargs):
    return get_openmx_template().new(**kwargs)


class DataDrivenCalculator(FileIOCalculator):
    implemented_properties = None
    command = None

    def __init__(self, template, **kwargs):
        self.template = template
        self.cache = None

        FileIOCalculator.__init__(self, label='hello',
                                  command=template.command,
                                  **kwargs)

    @property
    def implemented_properties(self):
        return self.template.implemented_properties

    @property
    def name(self):
        return self.template.name

    def set(self, **kwargs):
        changed_parameters = FileIOCalculator.set(self, **kwargs)
        if changed_parameters:
            self.reset()

    def write_input(self, atoms, properties=None, system_changes=None):
        FileIOCalculator.write_input(self, atoms, properties, system_changes)
        fmt = ioformats[self.template.input_format]
        write(self.template.input_file, atoms, format=fmt.name,
              **self.parameters)

    def read_results(self):
        fmt = ioformats[self.template.output_format]
        output = read(self.template.output_file, format=fmt.name)
        self.cache = output.calc
        self.results = output.calc.results

    def get_fermi_level(self):
        if self.cache is None:
            raise PropertyNotPresent(error_template % 'Fermi level')
        return self.cache.get_fermi_level()

    def get_ibz_k_points(self):
        if self.cache is None:
            raise PropertyNotPresent(error_template % 'IBZ k-points')
        ibzkpts = self.cache.get_ibz_k_points()
        if ibzkpts is None:
            warnings.warn(warn_template % 'IBZ k-points')
        return ibzkpts

    def get_k_point_weights(self):
        if self.cache is None:
            raise PropertyNotPresent(error_template % 'K-point weights')
        k_point_weights = self.cache.get_k_point_weights()
        if k_point_weights is None:
            warnings.warn(warn_template % 'K-point weights')
        return k_point_weights

    def get_eigenvalues(self, **kwargs):
        if self.cache is None:
            raise PropertyNotPresent(error_template % 'Eigenvalues')
        eigenvalues = self.cache.get_eigenvalues(**kwargs)
        if eigenvalues is None:
            warnings.warn(warn_template % 'Eigenvalues')
        return eigenvalues

    def get_number_of_spins(self):
        if self.cache is None:
            raise PropertyNotPresent(error_template % 'Number of spins')
        nspins = self.cache.get_number_of_spins()
        if nspins is None:
            warnings.warn(warn_template % 'Number of spins')
        return nspins
