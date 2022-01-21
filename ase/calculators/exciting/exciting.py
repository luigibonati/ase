"""ASE Calculator for the ground state exciting DFT code.

Exciting calculator class in this file allow for writing exciting input
files using ASE Atoms object that allow for the compiled exciting binary
to run DFT on the geometry/material defined in the Atoms object. Also gives
access to developer to a lightweight parser (lighter weight than NOMAD or
the exciting parser in the exciting repository) to capture ground state
properties.
"""

from abc import ABC
import os
from pathlib import Path
from typing import Union, List, Optional, Mapping
from xml.etree import ElementTree as ET

import ase
import ase.io.exciting

from ase.calculators.genericfileio import (GenericFileIOCalculator, CalculatorTemplate)
from ase.calculators.exciting.runner import ExcitingRunner, SubprocessRunResults
from ase.calculators.calculator import InputError, PropertyNotImplementedError

try:
    __import__('excitingtools')
    from excitingtools.input.base_class import query_exciting_version
    from excitingtools.input.input import exciting_input_xml
    from excitingtools.input.ground_state import ExcitingGroundStateInput
    from excitingtools.input.structure import ExcitingStructure
except ModuleNotFoundError:
    message = """excitingtools must be installed. TODO Make available via wheels"""
    raise ModuleNotFoundError(message)


class ExcitingProfile:
    """Defines all quantities that are configurable (variable) for a given machine.

    This follows the generic pattern BUT currently not used by our calculator as:
       * species_path is part of the input file in exciting.
       * Only part of the profile used in the base class is the run method,
         which is part of the BinaryRunner class.
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
    parser = {'info.xml': ase.io.exciting.parse_info_out_xml}
    output_names = list(parser)
    implemented_properties = ['energy', 'tforce']

    def __init__(self):
        """Initialise with constant class attributes.

        Args:
            program_name: The DFT program, should always be exciting.
            implemented_properties: What properties should exciting calculate/read from output.
        """
        super().__init__(self.program_name, self.implemented_properties)

    @staticmethod
    def _require_forces(input_parameters: Union[dict, ExcitingGroundStateInput]):
        """Expect that ASE always wants forces, so enforce the setting in input_parameters.

        Args:
            input_parameters: exciting ground state input parameters, either as a dictionary or ExcitingGroundStateInput
        Returns:
            input_parameters: Ground state input parameters, with "compute forces" set to true
        """
        if isinstance(input_parameters, dict):
            input_parameters['tforce'] = True
        elif isinstance(input_parameters, ExcitingGroundStateInput):
            input_parameters.__dict__['tforce'] = True
        else:
            raise ValueError('exciting input_parameters must be type [dict, ExcitingGroundStateInput]')
        return input_parameters


    def write_input(self,
                    directory: Path,
                    atoms: ase.Atoms,
                    input_parameters: dict,
                    properties):
        """Write an exciting input.xml file based on the input args.

        TODO(dts, AlexB): Figure out why previously this function had a properties argument
            called properties: List[str]. I assume this was to target certain properties
            but I thought ground state calcs will all get the same default properties.
            (Alex) Added it back in so API is consistent
        TODO(dts, Alex) Consider correct way to pass input XML's title

        Args:
            directory: Directory in which to run calculator.
            atoms: ASE atoms object.
            input_parameters: exciting ground state input parameters, in a dictionary.
            Expect species_path, title and ground_state data, either in an object or as dict
            properties: ADD ME
        """
        assert set(input_parameters.keys()) == {'title', 'species_path', 'ground_state_input'}, \
            'Keys should be defined by ExcitingGroundState calculator'

        file_name = directory / 'input.xml'
        species_path = input_parameters.pop('species_path')
        structure = ExcitingStructure(atoms, species_path=species_path)
        title = input_parameters.pop('title')

        input_parameters = self._require_forces(input_parameters['ground_state_input'])
        if isinstance(input_parameters, dict):
            ground_state = ExcitingGroundStateInput(**input_parameters)
        else:
            ground_state = input_parameters

        input_xml: ET.ElementTree = exciting_input_xml(structure, title=title, groundstate=ground_state)
        input_xml.write(file_name)

    def execute(self, directory, exciting_calculation: ExcitingRunner) -> SubprocessRunResults:
        """Given an exciting calculation profile, execute the calculation.
        TODO(all): Not working.

        Args:
            directory: Directory in which to execute the calculator
            exciting_calculation: Base method `execute` expects a profile, however it is simply
                used to execute the program, therefore we just pass an ExcitingRunner.

        Returns:
            Results of subprocess.run
        """
        return exciting_calculation.run(directory)

    def read_results(self, directory: Path) -> Mapping[str, any]:
        """Parse results from each ground state output file.

          Note we allow for the ability for there to be multiple output files.

          Args:
              directory: Directory path to output file from exciting simulation.
          Returns:
              Dictionary (technically abc.Mapping) containing important output properties.
        """
        results = {}
        for file_name in self.output_names:
            full_file_path = directory / file_name
            result: dict = self.parser[file_name](full_file_path)
            results.update(result)
        return results


class ExcitingGroundStateResults:
    """Exciting Ground State Results."""
    def __init__(self, results: dict) -> None:
        self.results = results
        self.completed = self.calculation_completed()
        self.converged = self.calculation_converged()

    def calculation_completed(self) -> bool:
        """Check if calculation is complete."""
        # TODO(Alex) This will be returned by the runner object - need to propagate the result to here
        return False

    def calculation_converged(self) -> bool:
        """Check if calculation is converged."""
        # TODO(Alex) This needs to parsed from INFO.OUT (or info.xml?)
        #  First check is that this file is written
        return False

    def potential_energy(self) -> float:
        """Return potential energy of system.

        TODO(Alex): Make the description of potential energy in physics terms more clear.
        """
        # TODO(Alex) We should a common list of keys somewhere
        # such that parser -> results -> getters are consistent
        return self.results['potential_energy'].copy()

    def forces(self):
        """Return forces present on the system.

        TODO(Dan): Add a bit more description here on how the forces are defiend
            and returned. Right now I can't find forces in exciting test info.xml
            files. So it looks like we shouldn't be parsing this all the time:
            https://git.physik.hu-berlin.de/sol/exciting/-/blob/development/test/test_farm/groundstate/LDA_PW-PbTiO3/ref/info.xml

        Returns:
            A copy of the forces present in the system.
        """
        # TODO(Alex) We should a common list of keys somewhere
        return self.results['forces'].copy()

    def stress(self):
        """Get the stress on the system.

        Right now exciting does not yet calculate the stress on the system so
        this won't work for the time being."""
        raise PropertyNotImplementedError


class ExcitingGroundState(GenericFileIOCalculator):
    """Class for the ground state calculation.

    Base class implements the calculate method.
    TODO(Alex) Revise the documentation

    Must supply to the constructor:
        * runner: ExcitingRunner: This should be a `profile` which is the machine-specific
          settings (run command, species paths, etc), however in exciting's case, this doesn't
          make sense. The species path is specified in the input file, and should therefore be
          an attribute of ExcitingInput. The only other machine-specific setting is the BinaryRunner.
          Furthermore, in GenericFileIOCalculator.calculate, profile is only used to provide a
          run method. We therefore pass the BinaryRunner in the place of a profile.

        * exciting_input ExcitingInput: exciting input parameters, including the species path
          BUT minus the structure, which is defined in ASE's Atoms object.

        * directory: Directory in which to run/execute the job.

    To the calculate method args:
        * atoms           ASE atoms object
        * properties      Output properties, as defined in a CalculatorTemplate
        * system_changes  ?

    via self:
        * directory      Directory in which to run/execute the job.
        * template       Specialised exciting template, containing write_input, execute and read_results methods
        * parameters     exciting_input. Can be defined as a class or Object. Responsibility
                         of the specialised write method.

    List of keyword inputs from the old calculator:
        species_path, kpts, autormt, tshift
    """
    default_title = 'ASE-generated input'

    def __init__(self, *,
                 runner: ExcitingRunner,
                 ground_state_input: Union[dict, ExcitingGroundStateInput],
                 directory='./',
                 species_path='./',
                 title=default_title):

        self.runner = runner
        # Data to be passed to ExcitingGroundStateTemplate.write_input(..., input_parameters, ...)
        # Structure not included, as it's passed when one calls .calculate method directly
        self.exciting_inputs = {'title': title, 'species_path': species_path, 'ground_state_input': ground_state_input}
        self.directory = directory
        self.template = ExcitingGroundStateTemplate()

        super().__init__(profile=runner,
                         template=self.template,
                         parameters=self.exciting_inputs,
                         directory=directory
                         )

    def calculate(self,
                  atoms: ase.Atoms,
                  properties: Optional[List[str]] = None,
                  system_changes=None) -> ExcitingGroundStateResults:
        """Run an exciting calculation and capture the results in ExcitingGroundStateResults object.

        TODO(Alex) Would rather remove properties from our calculate API.
        TODO(All) Document
        """
        if properties is None:
            properties = self.template.implemented_properties
        super().calculate(atoms, properties, system_changes)
        return ExcitingGroundStateResults(self.results)
