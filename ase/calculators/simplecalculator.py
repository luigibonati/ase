import warnings
from ase.io import read, write
from ase.io.formats import ioformats
from ase.calculators.calculator import FileIOCalculator, PropertyNotPresent


class CalculatorTemplate:
    def __init__(self):
        pass


def get_espresso_template():
    template = CalculatorTemplate()
    template.name = 'espresso'
    template.implemented_properties = ['energy', 'forces', 'stress', 'magmoms']
    template.command = 'pw.x -in PREFIX.pwi > PREFIX.pwo'
    template.input_format = 'espresso-in'
    return template

espresso_template = get_espresso_template()

def new_espresso(**kwargs):
    calc = CalculatorIndependentCalculator(espresso_template, **kwargs)
    return calc

class CalculatorIndependentCalculator(FileIOCalculator):
    implemented_properties = None
    #implemented_properties = ['energy', 'forces', 'stress', 'magmoms']
    command = None

    def __init__(self, template, **kwargs):
        self.template = template
        self.calc = None

        #self.name = self.template.name  # XXX
        FileIOCalculator.__init__(self, label='hello',
                                  command=template.command,
                                  **kwargs)

    @property
    def implemented_properties(self):
        return self.template.implemented_properties

    @property
    def name(self):
        return self.template.name

    #@property
    #def command(self):
    #    return self.template.command

    def set(self, **kwargs):
        changed_parameters = FileIOCalculator.set(self, **kwargs)
        if changed_parameters:
            self.reset()

    def write_input(self, atoms, properties=None, system_changes=None):
        FileIOCalculator.write_input(self, atoms, properties, system_changes)
        fmt = ioformats[self.template.input_format]
        ext = fmt.extensions[0]
        fname = self.label + '.' + ext
        write(fname, atoms, format=fmt.name, **self.parameters)

    def read_results(self):
        output = read(self.label + '.pwo')
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
