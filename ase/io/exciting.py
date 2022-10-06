"""This is the implementation of the exciting I/O functions.

The main roles these functions do is write exciting ground state
input files and read exciting ground state ouput files.

Right now these functions all written without a class to wrap them. This
could change in the future but was done to make things simpler.

These functions are primarily called by the exciting caculator in
ase/calculators/exciting/exciting.py.

See the correpsonding test file in ase/test/io/test_exciting.py.

Plan is to add parsing of eigenvalues in the next iteration using
excitingtools.exciting_dict_parsers.groundstate_parser.parse_eigval
"""
import logging
from pathlib import Path
from typing import Dict
from xml.etree import ElementTree as ET

import ase


LOGGER = logging.getLogger(__name__)

try:
    __import__('excitingtools')
    from excitingtools.exciting_dict_parsers.groundstate_parser import parse_info_out
    from excitingtools.input.input_xml import exciting_input_xml
    from excitingtools.input.ground_state import ExcitingGroundStateInput
    from excitingtools.input.structure import ExcitingStructure
except ModuleNotFoundError:
    MESSAGE = (
        "excitingtools must be installed with pip install excitingtools for"
        " the exciting io to work.")
    LOGGER.warn(MESSAGE)


def parse_output(info_out_file_path):
    """Parse exciting INFO.OUT output file using excitingtools."""
    # Check for the file:
    if not Path(info_out_file_path).is_file():
        raise FileNotFoundError
    return parse_info_out(info_out_file_path)

def write_input_xml_file(
        file_name, atoms: ase.Atoms, input_parameters: Dict,
        species_path, title = None):
    """Write input xml file for exciting calculation.

    Args:
        file_name: where to save the input xml file.
        atoms: ASE Atoms object.
        input_paramaters: Ground state parameters to affect exciting calc.
    """
    if isinstance(input_parameters, dict):
        ground_state = ExcitingGroundStateInput(**input_parameters)
    else:
        ground_state = input_parameters

    structure = ExcitingStructure(atoms, species_path=species_path)

    input_xml = exciting_input_xml(
        structure=structure, title=title, groundstate=ground_state)
    input_xml = ET.ElementTree(input_xml)
    with open (file_name, "wb") as file_handle:
        input_xml.write(file_handle)
