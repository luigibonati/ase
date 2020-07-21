from ase import Atoms
from ase.io import write
from ase.io.bagel import get_bagel_results
from ase.calculators.calculator import FileIOCalculator


class BAGEL(FileIOCalculator):
    implemented_properties = ['energy', 'forces']
    command = 'BAGEL PREFIX.bgi > PREFIX.bgo'
    discard_results_on_any_change = True
    default_parameters = {'state': 0}

    def __init__(
        self,
        restart: bool = None,
        ignore_bad_restart_file: bool = False,
        label: str = 'bagel',
        atoms: Atoms = None,
        command: str = None,
        **kwargs
    ) -> None:
        FileIOCalculator.__init__(self, restart, ignore_bad_restart_file,
                                  label, atoms, command, **kwargs)

    def write_input(self, atoms, properties=None, system_changes=None):
        FileIOCalculator.write_input(self, atoms, properties, system_changes)
        write(self.label + '.bgi', atoms, properties=properties,
              format='bagel-in', **self.parameters)

    def read_results(self):
        output = get_bagel_results(
            self.label + '.bgo',
            self.directory,
            state=self.parameters.state
        )

        self.calc = output.calc
        self.results = self.calc.results
