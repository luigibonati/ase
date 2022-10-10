"""ASE Calculator for the ground state exciting DFT code.

Exciting calculator class in this file allow for writing exciting input
files using ASE Atoms object that allow for the compiled exciting binary
to run DFT on the geometry/material defined in the Atoms object. Also gives
access to developer to a lightweight parser (lighter weight than NOMAD or
the exciting parser in the exciting repository) to capture ground state
properties.

Note: excitingtools must be installed using `pip install excitingtools` to
work.
"""

from abc import ABC
import logging
from os import PathLike
from pathlib import Path
from typing import Union, List, Optional, Mapping

import ase.io.exciting

from ase.calculators.genericfileio import (
    GenericFileIOCalculator, CalculatorTemplate)
from ase.calculators.exciting.runner import ExcitingRunner, SubprocessRunResults
from ase.calculators.calculator import PropertyNotImplementedError


LOGGER = logging.getLogger(__name__)


try:
    __import__('excitingtools')
    from excitingtools.input.base_class import query_exciting_version
    from excitingtools.input.ground_state import ExcitingGroundStateInput
except ModuleNotFoundError:
    MESSAGE = (
        "excitingtools must be installed with pip install excitingtools for"
        " the exciting io to work.")
    LOGGER.warn(MESSAGE)


class ExcitingProfile:
    """Defines all quantities that are configurable for a given machine.

    Follows the generic pattern BUT currently not used by our calculator as:
       * species_path is part of the input file in exciting.
       * OnlyTypo fix part of the profile used in the base class is the run
         method, which is part of the BinaryRunner class.
    """
    def __init__(self, exciting_root, species_path):
        self.species_path = species_path
        self.version = query_exciting_version(exciting_root)


class ExcitingGroundStateTemplate(CalculatorTemplate, ABC):
    """Template for Ground State Exciting Calculator

    Abstract methods inherited from the base class:
        * write_input
        * execute
        * read_results
    """
    program_name = 'exciting'
    parser = {'info.xml': ase.io.exciting.parse_info_out}
    output_names = list(parser)
    # Use frozenset since the CalculatorTemplate enforces it.
    implemented_properties = frozenset(['energy', 'tforce'])

    def __init__(self, binary_runner = None):
        """Initialise with constant class attributes.

        :param program_name: The DFT program, should always be exciting.
        :param implemented_properties: What properties should exciting
            calculate/read from output.
        """
        super().__init__(self.program_name, self.implemented_properties)
        self.binary_runner = binary_runner

    @staticmethod
    def _require_forces(input_parameters: Union[dict, ExcitingGroundStateInput]):
        """Expect ASE always wants forces, enforce setting in input_parameters.

        :param input_parameters: exciting ground state input parameters, either as
                a dictionary or ExcitingGroundStateInput.
        :return: Ground state input parameters, with "compute
                forces" set to true.
        """
        if isinstance(input_parameters, dict):
            input_parameters['tforce'] = True
        elif isinstance(input_parameters, ExcitingGroundStateInput):
            input_parameters.__dict__['tforce'] = True
        else:
            raise ValueError(
                'exciting input_parameters must be type'
                ' [dict, ExcitingGroundStateInput]')
        return input_parameters

    def write_input(self,
                    directory: PathLike,
                    atoms: ase.Atoms,
                    parameters: dict,
                    properties=None):
        """Write an exciting input.xml file based on the input args.

        :param directory: Directory in which to run calculator.
        :param atoms: ASE atoms object.
        :param parameters: exciting ground state input parameters, in a
            dictionary. Expect species_path, title and ground_state data,
            either in an object or as dict.
        :param properties: Currently, unused. Base method's API expects the
            physical properties expected from a ground state
            calculation, for example energies and forces.
        """
        del properties  # Unused but kept for API consistency.
        assert set(parameters.keys()) == {
            'title', 'species_path', 'ground_state_input'}, \
            'Keys should be defined by ExcitingGroundState calculator'
        file_name = Path(directory) / 'input.xml'
        species_path = parameters.pop('species_path')
        title = parameters.pop('title')

        ase.io.exciting.write_input_xml_file(
            file_name, atoms, parameters['ground_state_input'],
            species_path, title)

    def execute(
            self, directory: PathLike,
            profile=None) -> SubprocessRunResults:
        """Given an exciting calculation profile, execute the calculation.

        :param directory: Directory in which to execute the calculator
            exciting_calculation: Base method `execute` expects a profile,
            however it is simply used to execute the program, therefore we
            just pass an ExcitingRunner.
        :param profile: This name comes from the superclass CalculatorTemplate.
                We don't use the variable here and instead use the exciting
                binary runner instead to call the run command to execute the
                simulation.

        :return: Results of the subprocess.run command.
        """
        del profile  # Unused, but kept for API consistency.
        if self.binary_runner is None:
            raise ValueError(
                'Binary runner attribute of ExcitingGroundStateTemplate object'
                ' is None')
        return self.binary_runner.run(directory)

    def read_results(self, directory: PathLike) -> Mapping[str, str]:
        """Parse results from each ground state output file.
        
        Note we allow for the ability for there to be multiple output files.

        :param directory: Directory path to output file from exciting simulation.
        :return: Dictionary containing important output properties.
        """
        results = {}
        for file_name in self.output_names:
            full_file_path = Path(directory) / file_name
            result: dict = self.parser[file_name](full_file_path)
            results.update(result)
        return results


class ExcitingGroundStateResults:
    """Exciting Ground State Results."""
    def __init__(self, results: dict) -> None:
        self.results = results
        self.final_scl_iteration = list(results["scl"].keys())[-1]

    def total_energy(self) -> float:
        """Return total energy of system."""
        # TODO(Alex) We should a common list of keys somewhere
        # such that parser -> results -> getters are consistent
        return float(
            self.results['scl'][self.final_scl_iteration][
                'Total energy'])

    def band_gap(self) -> float:
        """Return the estimated fundamental gap from the exciting sim."""
        return float(
            self.results['scl'][self.final_scl_iteration][
                'Estimated fundamental gap'])

    def forces(self):
        """Return forces present on the system.

        Currently, not all exciting simulations return forces. We leave this
        definition for future revisions.
        """
        raise PropertyNotImplementedError

    def stress(self):
        """Get the stress on the system.

        Right now exciting does not yet calculate the stress on the system so
        this won't work for the time being.
        """
        raise PropertyNotImplementedError


class ExcitingGroundStateCalculator(GenericFileIOCalculator):
    """Class for the ground state calculation.

    :param runner: Binary runner that will execute an exciting calculation and
        return a result.
    :param ground_state_input: dictionary of ground state settings for example
        {'rgkmax': 8.0, 'autormt': True} or an object of type
        ExcitingGroundStateInput.
    :param directory: Directory in which to run the job.
    :param species_path: Path to the location of exciting's species files.
    :param title: job name written to input.xml

    :return: Results returned from running the calculate method.


    Typical usage:

    gs_calculator = ExcitingGroundState(runner, ground_state_input)

    results: ExcitingGroundStateResults = gs_calculator.calculate(
            atoms: Atoms)
    """
    def __init__(self, *,
                 runner: ExcitingRunner,
                 ground_state_input: Union[dict, ExcitingGroundStateInput],
                 directory='./',
                 species_path='./',
                 title='ASE-generated input'):

        self.runner = runner
        # Package data to be passed to
        # ExcitingGroundStateTemplate.write_input(..., input_parameters, ...)
        # Structure not included, as it's passed when one calls .calculate
        # method directly
        self.exciting_inputs = {
            'title': title, 'species_path': species_path,
            'ground_state_input': ground_state_input}
        self.directory = Path(
            directory) if isinstance(directory, str) else directory
        self.template = ExcitingGroundStateTemplate(binary_runner=runner)

        # GenericFileIOCalculator expects a `profile`
        # containing machine-specific settings, however, in exciting's case,
        # the species file are defined in the input XML (hence passed in the
        # parameters argument) and the only other machine-specific setting is
        # the BinaryRunner. Furthermore, in GenericFileIOCalculator.calculate,
        # profile is only used to provide a run method. We therefore pass the
        # BinaryRunner in the place of a profile.
        super().__init__(profile=runner,
                         template=self.template,
                         parameters=self.exciting_inputs,
                         directory=directory)

    def calculate(self,
                  atoms: ase.Atoms,
                  properties: Optional[List[str]] = None,
                  system_changes=None) -> ExcitingGroundStateResults:
        """Run an exciting calculation and capture the results.

        Results are captured in a ExcitingGroundStateResults object.

        The base class implements the method.

        Args:
            atoms: ase.Atoms object containing atomic positions and lattice
                vectors.
            properties: Properties expected by the calculation. This is passed
                to the template's write_input method, where it is not used. We
                include it here to satisfy the API.
            system_changes: This is unused by the base method, so we only
                include to maintain a consistent API.
        Returns:
            ExcitingGroundStateResults object.
        """
        if properties is None:
            properties = self.template.implemented_properties
        # The base class GenericFileIOCalculator writes the exciting input
        # based on ExcitingGroundStateTemplate's write input method and passes
        # the self.input_parameters to the write_input. Then it calls
        # the execute method of ExcitingGroundStateTemplate which ultimately
        # uses the binary runner and then read's reasults.
        super().calculate(atoms, properties, system_changes)
        return ExcitingGroundStateResults(results=self.results)
