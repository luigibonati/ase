"""Test file for exciting file input and output methods."""
import pytest
import tempfile  # Used to create temporary directories for tests.
import io
import numpy as np
import xml.etree.ElementTree as ET

import ase
import ase.io.exciting
from ase.units import Bohr


@pytest.fixture
def nitrogen_trioxide_atoms():
    return ase.Atoms('NO3',
                     cell=[[2, 2, 0], [0, 4, 0], [0, 0, 6]],
                     positions=[(0, 0, 0), (1, 3, 0),
                                (0, 0, 1), (0.5, 0.5, 0.5)],
                     pbc=True)

def test_initialize_element_tree():
    """Test initializing an element tree."""
    initialized_element_tree = ase.io.exciting.initialise_input_xml(title_text='title_test')
    # Check that a title has been set on the xml object.
    assert initialized_element_tree.find('title').text == 'title_test'
    # Check that structure and structure.crystal have been added.
    assert initialized_element_tree.find('structure') is not None 
    assert initialized_element_tree.find('structure/crystal') is not None

def test_structure_element_tree(nitrogen_trioxide_atoms):
    """Test adding an ASE Atoms object to the XML element tree."""
    # Check that the tree is initialized correctly before proceeding.
    initialized_element_tree = ase.io.exciting.initialise_input_xml()
    element_tree = ase.io.exciting.add_atoms_to_structure_element_tree(
        structure=initialized_element_tree, atoms=nitrogen_trioxide_atoms)
    assert element_tree.tag == 'input'
    # Check the chidren has been added correctly.
    expected_children_tags = ['title', 'structure', 'species', 'species', 'species']
    for count, child in enumerate(element_tree):
        assert child.tag == expected_children_tags[count]
    # TODO(Alex) Shouldn't convert in the reference data - should be correct by inspection.
    # Check the lattice vectors have been added correctly to the XML object.
    expected_vectors = [[2 / Bohr, 2 / Bohr, 0], [0, 4 / Bohr, 0], [0, 0, 6 / Bohr]]
    basis_vector_list = element_tree.findall('./structure/crystal/basevect')
    for i in range(len(basis_vector_list)):
        float_vector = [float(x) for x in basis_vector_list[i].text.split()]
        assert len(float_vector) == len(expected_vectors[i])
        assert all([np.round(a - b, 14) == 0 for a, b in zip(float_vector, expected_vectors[i])])
    # Ensure the species data has been added correctly.
    expected_chemical_symbols = ['N', 'O']
    species = element_tree.findall('./structure/species')
    for i in range(len(species)):
        assert species[i].get('chemicalSymbol') == expected_chemical_symbols[i]
    # Ensure the coordinates of the atoms in the unit cell is correct.
    expected_coords = [[0, 0, 0], [0.5, 0.5, 0], [0, 0, 1 / 6], [0.25, 0, 1 / 12]]
    coords_list = element_tree.findall('./structure/species/atom')
    for i in range(len(coords_list)):
        float_vector = [float(x) for x in coords_list[i].get('coord').split()]
        assert len(float_vector) == len(expected_coords[i])
        assert all([np.round(a - b, 14) == 0 for a, b in zip(float_vector, expected_coords[i])])


def test_dict_to_xml_adding_text():
    """Tests setting text of an element with dict_to_xml method."""
    element = ET.Element('root')
    dictionary = {'text()': 'test'}
    ase.io.exciting.dict_to_xml(dictionary, element)
    assert element.text == 'test'


def test_dict_to_xml_adding_attribute():
    """Tests adding an attribute with dict_to_xml method."""
    element = ET.Element('root')
    dictionary = {'number': '2'}
    ase.io.exciting.dict_to_xml(dictionary, element)
    assert element.attrib.get('number') == '2'


def test_dict_to_xml_list_in_pdict():
    """Tests dict_to_xml method with a list in pdict parameter."""
    element = ET.Element('root')
    dictionary = {'sub': [{'text()': 'test', 'number': '2'}]}
    ase.io.exciting.dict_to_xml(dictionary, element)
    assert isinstance(element.find('./sub'), ET.Element)
    assert element.find('./sub').text == 'test'
    assert element.find('./sub').attrib.get('number') == '2'


def test_dict_to_xml_dict_in_pdict_new_subelement():
    """Tests dict_to_xml method with a dict in pdict parameter by creating a new subelement."""
    element = ET.Element('root')
    dictionary = {'sub': {'number': '2'}}
    ase.io.exciting.dict_to_xml(dictionary, element)
    assert isinstance(element.find('./sub'), ET.Element)
    assert element.find('./sub').attrib.get('number') == '2'


def test_dict_to_xml_dict_in_pdict_add_to_subelement():
    """Tests dict_to_xml method with a dict in pdict parameter by adding to an existing subelement."""
    element = ET.Element('root')
    ET.SubElement(element, 'sub')
    ET.SubElement(element, 'sub')
    dictionary = {'sub': {'number': '2'}}
    ase.io.exciting.dict_to_xml(dictionary, element)
    sub_elements = element.findall('./sub')
    assert len(sub_elements) == 2
    assert sub_elements[0].attrib.get('number') == '2'
    assert len(sub_elements[1].keys()) == 0


def test_dict_to_xml_wrong_arguments():
    """Tests dict_to_xml method if the arguments are not supported."""
    with pytest.raises(TypeError, match='cannot deal with'):
        element = ET.Element('root')
        dictionary = {'sub': 1}
        ase.io.exciting.dict_to_xml(dictionary, element)


def test_parse_info_out_xml():
    """Tests parse method if info.xml does not exist."""
    # RuntimeError should be raised when the xml file doesn't exist that we want to parse.
    with pytest.raises(RuntimeError):
        ase.io.exciting.parse_info_out_xml('input_not_exist.xml')


