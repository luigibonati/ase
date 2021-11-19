"""Test file for exciting file input and output methods."""
import pytest
import tempfile  # Used to create temporary directories for tests.
import unittest

import ase
import ase.io.exciting
from ase.build import bulk
from ase.io import read, write
from ase.units import Bohr
import numpy as np
import xml.etree.ElementTree as ET

'''
def test_exciting_io():
    """Old test that should be depracated."""
    atoms = ase.Atoms('N3O',
                  cell=[3, 4, 5],
                  positions=[(0, 0, 0), (1, 0, 0),
                             (0, 0, 1), (0.5, 0.5, 0.5)],
                  pbc=True)

    write('input.xml', atoms)
    atoms2 = read('input.xml')

    assert all(atoms.symbols == atoms2.symbols)
    assert atoms.cell[:] == pytest.approx(atoms2.cell[:])
    assert atoms.positions == pytest.approx(atoms2.positions)


def test_write(self, calculator):
    """Test the write method if it has to create a directory."""
    atoms = ase.Atoms('N3O',
                      cell=[3, 4, 5],
                      positions=[(0, 0, 0), (1, 0, 0),
                                 (0, 0, 1), (0.5, 0.5, 0.5)],
                      pbc=True)
    calculator.dir = calculator.dir + '/test'
    expected_output_file_path = calculator.dir + '/input.xml'
    calculator.paramdict = {'number': '2'}
    calculator.write_input(bulk('Fe'))
    assert expected_output_file_path == output_file_path
'''


def test_atoms_to_etree(nitrogen_trioxide_atoms):
    element_tree = ase.io.exciting.atoms_to_etree(nitrogen_trioxide_atoms)
    assert element_tree.tag == 'input'
    expected_children_tags = ['title', 'structure']
    for count, child in enumerate(element_tree):
        assert child.tag == expected_children_tags[count]

    expected_vectors = [[2 / Bohr, 2 / Bohr, 0], [0, 4 / Bohr, 0], [0, 0, 6 / Bohr]]
    basis_vector_list = element_tree.findall('./structure/crystal/basevect')
    for i in range(len(basis_vector_list)):
        float_vector = [float(x) for x in basis_vector_list[i].text.split()]
        assert len(float_vector) == len(expected_vectors[i])
        assert all([np.round(a-b, 14) == 0 for a, b in zip(float_vector, expected_vectors[i])])

    expected_chemical_symbols = ['N', 'O']
    species = element_tree.findall('./structure/species')
    for i in range(len(species)):
        assert species[i].get('chemicalSymbol') == expected_chemical_symbols[i]

    expected_coords = [[0, 0, 0], [0.5, 0.75, 0], [0, 0, 1/6], [0.25, 0, 1/12]]
    coords_list = element_tree.findall('./structure/species/atom')
    for i in range(len(coords_list)):
        float_vector = [float(x) for x in coords_list[i].get('coord').split()]
        assert len(float_vector) == len(expected_coords[i])
        assert all([np.round(a-b, 14) == 0 for a, b in zip(float_vector, expected_coords[i])])


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


def test_add_attributes_to_element_tree(nitrogen_trioxide_atoms):
    species_path = 'fake_path'
    element_tree = ase.io.exciting.add_attributes_to_element_tree(atoms=nitrogen_trioxide_atoms, autormt=False,
                                                                  species_path=species_path, tshift=True,
                                                                  param_dict={'groundstate': {'nempty': '2'}})
    expected_chemical_symbols = ['N', 'O']
    species = element_tree.findall('./structure/species')
    for i in range(len(species)):
        assert species[i].get('chemicalSymbol') == expected_chemical_symbols[i]
    assert element_tree.findall('./structure')[0].get('tshift') == 'true'
    assert element_tree.findall('./structure')[0].get('autormt') == 'false'
    assert element_tree.findall('./groundstate')[0].get('nempty') == '2'
    assert element_tree.findall('./structure')[0].get('speciespath') == species_path


@pytest.fixture
def nitrogen_trioxide_atoms():
    return ase.Atoms('NO3',
                     cell=[[2, 2, 0], [0, 4, 0], [0, 0, 6]],
                     positions=[(0, 0, 0), (1, 0, 0),
                                (0, 0, 1), (0.5, 0.5, 0.5)],
                     pbc=True)
