# """ASE Calculator for the exciting DFT code."""

import os
import ase
from typing import Union, List, Optional
from pathlib import Path
import subprocess

from ase.calculators.genericfileio import (GenericFileIOCalculator, CalculatorTemplate)


# TODO(Alex) Move
class SubprocessRunResults:
    def __init__(self, stdout, stderr, return_code: int):
        self.stdout = stdout
        self.stderr = stderr
        self.return_code = return_code
        self.success = return_code == 0


# TODO(Alex) Move
class SimpleBinaryRunner:
    """
    Compose a run command and run a binary
    # TODO(Alex) Remove all defaults
    """

    def __init__(self,
                 binary: str,
                 run_cmd: Optional[List[str]],
                 omp_num_threads: Optional[int],
                 args: Optional[List[str]] = [''],
                 time_out: Optional[int] = 600
                 ) -> None:
        """

        :param List[str] run_cmd: List of run commands, for example:
          * For serial: ['./']
          * For MPI:   ['mpirun', '-np', '2']

        """
        self.binary = binary
        self.run_cmd = run_cmd
        self.omp_num_threads = omp_num_threads
        self.args = args
        self.time_out = time_out

        assert omp_num_threads > 0, "Number of OMP threads must be > 0"

        try:
            i = run_cmd.index('np')
            mpi_processes = eval(run_cmd[i + 1])
            assert type(mpi_processes) == int, "Number of MPI processes should be an int"
            assert mpi_processes > 0, "Number of MPI processes must be > 0"
        except ValueError:
            pass

    def compose_execution_list(self) -> list:
        """
        Generate a complete list of strings to pass to subprocess.run to execute the calculation

        """
        return self.run_cmd + [self.binary] + self.args

    def run(self, directory: str, execution_list: Optional[list] = None) -> SubprocessRunResults:
        """
        Run a binary

        :param str directory: Directory in which to run the execute command.
        :param Optional[list] execution_list: List of arguments required by subprocess.run.
               Defaults to None.
        """

        # TODO(Alex) Change to directory

        if execution_list is None:
            execution_list = self.compose_execution_list()

        my_env = {**os.environ, "OMP_NUM_THREADS": str(self.omp_num_threads)}
        result = subprocess.run(execution_list,
                                env=my_env,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                timeout=self.time_out)

        return SubprocessRunResults(result.stdout, result.stderr, result.returncode)


class ExcitingRunner(SimpleBinaryRunner):
    """
    Execute an exciting calculation using a simple binary runner.

    Note, this class replaces "profile = ExcitingProfile", which is fucking
    meaningless, and in the case of EspressoProfile, mixed responsibilities.
    """

    # TODO(Alex) Check these
    binaries = ['exciting_serial', 'exciting_mpi', 'exciting_smp', 'exciting_mpiandsmp']
    input_name = 'input.xml'

    def __init__(self,
                 binary: str,
                 directory: str,
                 run_cmd: Optional[List[str]] = ['./'],
                 omp_num_threads: Optional[int] = 1,
                 args: Optional[List[str]] = [''],
                 time_out: Optional[int] = 600
                 ) -> None:
        # TODO(Alex) Pass directory
        assert binary in self.binaries, "binary string is not a valid choice"
        super().__init__(binary, run_cmd, omp_num_threads, args, time_out)


class ExcitingInput:
    pass


class ExcitingGroundStateInput(ExcitingInput):
    # STUB
    rgkmax: float = 6.0


class ExcitingGroundStateTemplate(CalculatorTemplate):
    """
    Template for Ground State Exciting Calculator
    """

    # TODO(Alex) Consider injecting our parsers here else use existing routines
    name = 'exciting'
    parser = {'info.xml': lambda file_name: {}}
    output_names = list(parser.keys())
    implemented_properties = ['energy', 'forces']

    def __init__(self):
        # Pass some stuff I'm not convinced we need but maintain the API
        super().__init__(self.name, self.implemented_properties)

    def write_input(self,
                    directory: Path,
                    atoms: ase.Atoms,
                    input_parameters: Union[dict, ExcitingGroundStateInput],
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

    # TODO(Alex) The generic calculate method should return SubprocessRunResults,
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
                 directory='.'):
        super().__init__(profile=runner,
                         template=template,
                         parameters=parameters,
                         directory=directory
                         )

# How I envision it being called
# run_settings = ExcitingRunner('exciting_mpiandsmp',
#                               run_cmd = ['mpirun', '-np', '2'],
#                               omp_num_threads = 2,
#                               directory: './')
#
# gs_input = ExcitingGroundStateInput(stuff, fill_defaults=False)
# exciting_calculator = ExcitingGroundState(run_settings, gs_input, atoms=atoms)
# results = exciting_calculator.calculate()

