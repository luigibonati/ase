"""
ASE Calculator for the exciting DFT code

# TODO(Alex) File name is a place-holder
# Profile = all things configurable on a machine basis
# Query the version of exciting
# Add the species default path
# TODO(Alex) Extend, by adding path to species =>
# Either need to copy them OR specify in input.xml
"""
import ase
from typing import Union, List, Optional
from pathlib import Path

from ase.calculators.genericfileio import (GenericFileIOCalculator, CalculatorTemplate)
from ase.calculators.exciting.runner import ExcitingRunner, SubprocessRunResults


class ExcitingInput:
    """
    Base class for exciting inputs
    """
    pass


class ExcitingGroundStateTemplate(CalculatorTemplate):
    """
    Template for Ground State Exciting Calculator
    """

    # TODO(Alex) Consider injecting our parsers here else use existing routines
    name = 'exciting'
    parser = {'info.xml': lambda file_name: {}}
    output_names = list(parser)
    implemented_properties = ['energy', 'forces']

    def __init__(self):
        # Pass some stuff I'm not convinced we need but maintain the API
        super().__init__(self.name, self.implemented_properties)

    def write_input(self,
                    directory: Path,
                    atoms: ase.Atoms,
                    input_parameters: Union[dict, ExcitingInput],
                    properties):
        """
        TODO Compose this around free functions
        TODO(Alex) Note, no idea what properties are.
        """
        directory.mkdir(exist_ok=True, parents=True)
        # Convert exciting inputs and ase.Atoms into an xml string
        input_xml = 'Add functions to compose and return me'
        # Write the XML string to file
        return

    # TODO(Alex) The generic calculate method call to execute should return SubprocessRunResults,
    #            such that one can a) capture stdout cleanly and b) capture stderr, such that it can be
    #            returned in results
    @staticmethod
    def execute(directory: str, exciting_calculation: ExcitingRunner) -> SubprocessRunResults:
        """
        Given an exciting calculation profile, execute the calculation.

        :return SubprocessRunResults: Results of subprocess.run
        """
        return exciting_calculation.run(directory)

    def read_results(self, directory: Path) -> dict:
        """
        Parse results from each ground state output file
        """
        results = {}
        for file_name in self.output_names:
            full_file_path = directory / file_name
            result: dict = self.parser[file_name](full_file_path)
            results.update(result)


class Exciting(GenericFileIOCalculator):
    """
    Exciting Calculator Class.

    Only need to initialise, as the base class implements the calculate method.

    Must supply to the constructor:
        * Calculator Template - Fine, just write wrappers over this
           i.e. class ExcitingGroundState(Exciting)
               init(...atoms=None,..)
                 self.template = ExcitingGroundStateTemplate()
                 self.atoms = atoms
        * Runner: I think this should always be required
        * exciting input parameters (minus structure/atoms)
        * Run directory

    To the calculate method:
        * atoms (annoyingly cannot supply when initialising - see solution above)
        * exciting_input_parameters via self
        * profile (Binary runner) via self
        * properties = None
        * directory via self
    """

    def __init__(self, *,
                 template: CalculatorTemplate,
                 runner: ExcitingRunner,
                 parameters: ExcitingInput,
                 directory='./'):
        super().__init__(profile=runner,
                         template=template,
                         parameters=parameters,
                         directory=directory
                         )



inputs = {'rgkmax': 8, "kpts": (1,1,1)}
runner = ExcitingRunner(['mpirun', '-np', '2'], 'exciting_binary')

# Just conform to this
# exciting_calc = ExcitingGroundState(inputs, runner)
# exciting_calc.calculate(atoms)

# results: dict = exciting_calc.results
# forces = results['forces']
# forces = exciting_calc.get_forces()


# See what the atoms object contains:
# exciting_calc.update_atoms(atoms)

#



class ExcitingGroundState(Exciting):
    def __init__(self,
                 runner: ExcitingRunner,
                 parameters: ExcitingInput,
                 directory='./'):

        template = ExcitingGroundStateTemplate()
        super().__init__(template,
                         runner,
                         parameters,
                         directory=directory)









# How I envision it being called
# run_settings = ExcitingRunner('exciting_mpiandsmp',
#                               run_cmd = ['mpirun', '-np', '2'],
#                               omp_num_threads = 2,
#                               directory: './')
#
# gs_input = ExcitingInput(stuff, fill_defaults=False)
# exciting_calculator = ExcitingGroundState(run_settings, gs_input, atoms=atoms)
# results = exciting_calculator.calculate()

