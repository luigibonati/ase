import sys
from ase.io import read, write
from ase.io.formats import ioformats
from ase.calculators.calculator import FileIOCalculator


class SingleFileReader:
    def __init__(self, output_file, fmt):
        self.output_file = output_file
        self.fmt = fmt

    def read(self):
        output = read(self.output_file, format=self.fmt)
        cache = output.calc
        return cache


class CalculatorTemplate:
    def __init__(self, name, implemented_properties, command,
                 input_file, input_format, reader):
        self.name = name
        self.implemented_properties = implemented_properties
        self.command = command

        # Generalize: We need some kind of Writer and Reader
        # to handle multiple files at a time.
        self.input_file = input_file
        self.input_format = input_format
        self.reader = reader

    def __repr__(self):
        return 'CalculatorTemplate({})'.format(vars(self))

    def new(self, **kwargs):
        return GenericFileIOCalculator(template=self, **kwargs)


def get_espresso_template():
    from ase.calculators.espresso import Espresso
    infile = 'espresso.pwi'
    outfile = 'espresso.pwo'
    return CalculatorTemplate(
        name='espresso',
        implemented_properties=Espresso.implemented_properties,
        command='pw.x -in {} > {}'.format(infile, outfile),
        input_file=infile,
        #output_file=outfile,
        input_format='espresso-in',
        reader=SingleFileReader(outfile, 'espresso-out'))


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
        input_format='traj',
        reader=SingleFileReader(outfile, 'traj'))


def new_espresso(**kwargs):
    return get_espresso_template().new(**kwargs)


def new_emt(**kwargs):
    return get_emt_template().new(**kwargs)


class GenericFileIOCalculator(FileIOCalculator):
    command = None
    discard_results_on_any_change = True

    def __init__(self, template, **kwargs):
        self.template = template
        self.cache = None

        super().__init__(label=None,
                         command=template.command,
                         **kwargs)

    def __repr__(self):
        return '{}({})'.format(type(self).__name__, self.template.name)

    @property
    def implemented_properties(self):
        return self.template.implemented_properties

    @property
    def name(self):
        return self.template.name

    def write_input(self, atoms, properties, system_changes):
        super().write_input(atoms, properties, system_changes)
        fmt = ioformats[self.template.input_format]
        kwargs = self.parameters
        if 'properties' in fmt.write.__code__.co_varnames:
            kwargs = dict(kwargs)
            kwargs['properties'] = properties
        write(self.template.input_file, atoms, format=fmt.name,
              **kwargs)

    def read_results(self):
        self.cache = self.template.reader.read()

    @property
    def results(self):
        if self.cache is None:
            return {}
        return self.cache.results

    @results.setter
    def results(self, value):
        assert value == {}
        self.cache = None

    def get_fermi_level(self):
        return self.cache.get_fermi_level()

    def get_ibz_k_points(self):
        return self.cache.get_ibz_k_points()

    def get_k_point_weights(self):
        return self.cache.get_k_point_weights()

    def get_eigenvalues(self, **kwargs):
        return self.cache.get_eigenvalues(**kwargs)

    def get_number_of_spins(self):
        return self.cache.get_number_of_spins()
