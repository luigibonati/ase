"""ASE Calculator for the ground state exciting DFT code.

Exciting calculator class in this file allow for writing exciting input
files using ASE Atoms object that allow for the compiled exciting binary
to run DFT on the geometry/material defined in the Atoms object. Also gives
access to developor to a lightweight parser (lighter weight than NOMAD or
the exciting parser in the exciting repository) to capture ground state
properties.
"""

from abc import ABC
import os
from pathlib import Path
from typing import Union, List, Optional

import ase
import ase.io.exciting

from ase.calculators.genericfileio import (GenericFileIOCalculator, CalculatorTemplate)
from ase.calculators.exciting.runner import ExcitingRunner, SubprocessRunResults
from ase.calculators.calculator import InputError, PropertyNotImplementedError
from ase.calculators.exciting.input import query_exciting_version, ExcitingInput


class ExcitingProfile:
    """Defines all quantities that are configurable (variable) for a given machine.

    This follows the generic pattern BUT currently not used by our calculator as:
       * species_path is part of the input file in exciting.
       * Only part of the profiled used in the base class is the run method,
         which is part of the BinaryRunner class.
    """
    def __init__(self, exciting_root, species_path):
        self.species_path = species_path
        self.version = query_exciting_version(exciting_root)


class ExcitingGroundStateTemplate(CalculatorTemplate, ABC):
    """Template for Ground State Exciting Calculator,
    
    The class requires implementations for the methods defined in CalculatorTemplate.
    """

    program_name = 'exciting'
    # TODO(Alex) Add parsers here i.e.
    # parser = {'info.xml': Callable[str, dict]}
    # parser = {'info.xml': lambda file_name: {}}
    parser = {
        'info.xml': ase.io.exciting.parse_info_out_xml}
    output_names = list(parser)
    implemented_properties = ['energy']

    def __init__(self):
        """Initialise with constant class attributes.

        Args:
            program_name: The DFT program, should always be exciting.
            implmented_properties: What properties should exciting calculate/read from output.
        """
        super().__init__(self.program_name, self.implemented_properties)

    def write_input(self, directory: Path, atoms: ase.Atoms,
                    input_parameters: Union[dict, ExcitingInput]):
        """Write an exciting input.xml file based on the input args.

        TODO(dts): Figure out why previously this function had a properties argument
            called properties: List[str]. I assume this was to target certain properties
            but I thought ground state calcs will all get the same default properties.

        Args:
            directory: Directory in which to run calculator.
            atoms: ASE atoms object.
            input_parameters: exciting groundstate input parameters
        """
        root = ase.io.exciting.initialise_input_xml()
        structure = ase.io.exciting.add_atoms_to_structure_element_tree(
            structure=root, atoms=atoms)
        if 'forces' in list(self.implemented_properties):
            input_parameters['tforce'] = 'true'
        ase.io.exciting.dict_to_xml(input_parameters, root)

        with open(os.path.join(directory, 'input.xml'), "w") as fileobj:
            fileobj.write(ase.io.exciting.prettify(root))

    def execute(self, directory, exciting_calculation: ExcitingRunner) -> SubprocessRunResults:
        """Given an exciting calculation profile, execute the calculation.

        TODO(all): Not working.
        
        Method could be static, but maintaining API consistent with CalculatorTemplate

        Args:
            directory: Directory in which to execute the calculator
            exciting_calculation: Generic `execute` expects a profile, however it is simply
                used to execute the program, therefore we just pass an ExcitingRunner.
        
        Returns:
            Results of subprocess.run
        """
        return exciting_calculation.run(directory)

    def read_results(self, directory: Path) -> dict:
        """Parse results from each ground state output file.

        Note we allow for the ability for there to be multiple output files.

        Args:
            directory: Directory path to output file from exciting simulation.
        Returns:
            Dictionary containing important output properties.
        """
        results = {}
        for file_name in self.output_names:
            print(file_name)
            full_file_path = os.path.join(directory, file_name)
            print(full_file_path)
            result: dict = self.parser[file_name](full_file_path, self.implemented_properties)
            results.update(result)
        return results


def check_key_present(key, exciting_input: Union[ExcitingInput, dict]) -> bool:
    """Checks key is specified in the ExcitingInput object.

    One important use case is to check whether path to exciting species files is
    specified.

    Args:
        key: Specific key we're looking for in the exciting_input object.
        exciting_input: This object/dict contains much of the information we specify
            about how we want the exciting input file to look like.
    Returns:
        True/False whether species path is specified by ExcitingInput object.
    """
    if isinstance(exciting_input, ExcitingInput):
        keys = ExcitingInput.__dict__
    else:
        keys = list(exciting_input)

    # TODO FINISH ME

    return key in keys


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

    TODO(Alex) What methods do we need from our old calculator, and what exist in the base classes?

    TODO(Alex) We could support a specific set of keyword args, then use either a) Input dict/object
        or b) keywords
        
    List of keyword inputs from the old calculator:
        species_path, kpts, autormt, tshift
    """

    # Input options that must be present in the exciting_input
    required_inputs = ['species_path']

    def __init__(self, *,
                 runner: ExcitingRunner,
                 exciting_input: Union[ExcitingInput, dict],
                 directory='./'):

        self.runner = runner
        self.exciting_input = exciting_input
        self.directory = directory
        self.template = ExcitingGroundStateTemplate()

        for key in self.required_inputs:
            if not check_key_present(key, self.exciting_input):
                raise InputError(f'Key missing from exciting input: {key}')

        super().__init__(profile=runner,
                         template=self.template,
                         parameters=exciting_input,
                         directory=directory
                         )

    # TODO(Alex) Would rather remove properties from our calculate API.
    def calculate(self,
                  atoms: ase.Atoms,
                  properties: Optional[List[str]] = None,
                  system_changes=None) -> ExcitingGroundStateResults:
        """Run an exciting calculation and capture the results in ExcitingGroundStateResults object."""
        if properties is None:
            properties = self.template.implemented_properties
        super().calculate(atoms, properties, system_changes)
        return ExcitingGroundStateResults(self.results)

    # TODO(Alex) Note to remove once confirmed.
    # update method not copied. Calculator class stores atoms in the BaseCalculator
    # but we don't need any API to interact with this.
    # One can just pass an updated atoms object to .calculate(atoms)
    #
    # initialize not needed. atoms object injected into XML via the template's write
    # function
