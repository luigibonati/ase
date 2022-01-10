"""Test file for exciting file input and output methods."""

import os
import pytest

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


def test_parse_info_out_xml_file_does_not_exist():
    """Tests parse method if info.xml does not exist."""
    # RuntimeError should be raised when the xml file doesn't exist that we want to parse.
    with pytest.raises(ValueError) as excinfo:
        ase.io.exciting.parse_info_out_xml('input_not_exist.xml', implemented_properties=None)
    
    assert 'Output file input_not_exist.xml does not exist.' in str(excinfo.value)

def test_parse_info_out_xml_not_converged(tmp_path):
    """Tests parse method if info.xml shows the calc is not converged"""
    # ASE doesn't want us to store any other files for test, so instead let's write
    # a temporary XML file. We could write it with XML but easier to copy a good known
    # XML file as a string above and write it directly to a file.
    input_string = """<?xml version="1.0"?>
    <?xml-stylesheet href="http://xml.exciting-code.org/info.xsl" type="text/xsl"?>
    <info date="2020-12-10" time="20:05:40" versionhash="1775bff4453c84689fb848894a9224f155377cfc" title="PbTiO3">
    <groundstate status="unfinished">
    </groundstate>
    </info>
    """
    output_file_path = os.path.join(tmp_path, 'info.xml')
    with open(output_file_path, "w") as xml_file:
        xml_file.write(input_string)
    assert os.path.exists(output_file_path)  # Ensure file has been written.

    # RuntimeError should be raised when the xml file doesn't exist that we want to parse.
    with pytest.raises(RuntimeError) as excinfo:
        ase.io.exciting.parse_info_out_xml(output_file_path, implemented_properties=None)
    
    assert 'Calculation did not converge.' in str(excinfo.value)

def test_parse_info_out_xml_bogus_xml(tmp_path):
    """Tests parse method raises error when xml file is improperly formatted."""
    # ASE doesn't want us to store any other files for test, so instead let's write
    # a temporary XML file. We could write it with XML but easier to copy a good known
    # XML file as a string above and write it directly to a file.
    input_string = """<bogus><bogus>"""
    output_file_path = os.path.join(tmp_path, 'info.xml')
    with open(output_file_path, "w") as xml_file:
        xml_file.write(input_string)
    assert os.path.exists(output_file_path)  # Ensure file has been written.

    # RuntimeError should be raised when the xml file doesn't exist that we want to parse.
    with pytest.raises(ET.ParseError):
        ase.io.exciting.parse_info_out_xml(output_file_path, implemented_properties=None)

def test_parse_info_out_energy(tmp_path):
    # Grab the exciting info.xml from:
    # https://git.physik.hu-berlin.de/sol/exciting/-/blob/development/test/test_farm/groundstate/LDA_PW-PbTiO3/ref/info.xml
    # ASE doesn't want us to store any other files for test, so instead let's write
    # a temporary XML file. We could write it with XML but easier to copy a good known
    # XML file as a string above and write it directly to a file.
    input_string = """<?xml version="1.0"?>
    <?xml-stylesheet href="http://xml.exciting-code.org/info.xsl" type="text/xsl"?>
    <info date="2020-12-10" time="20:05:40" versionhash="1775bff4453c84689fb848894a9224f155377cfc" title="PbTiO3">
    <groundstate status="finished">
        <scl>
        <iter iteration="1" rms="0.240509096538276" rmslog10="-0.618868493088" deltae="21958.9331756557" deltaelog10="4.34161123709" chgdst="4.795296521919598E-002" chgdstlog10="-1.31918453263" fermidos="0.00000000000">
            <energies totalEnergy="-21958.9331757" fermiEnergy="0.276461898034" sum-of-eigenvalues="-13530.0092931" electronic-kinetic="25450.8254535" core-electron-kinetic="0.00000000000" Coulomb="-46966.0760553" Coulomb-potential="-38392.1732943" nuclear-nuclear="-1618.19059132" electron-nuclear="-52303.5976336" Hartree="6955.71216963" Madelung="-27769.9894081" xc-potential="-588.661452269" exchange="-431.539710366" correlation="-12.1428635285"/>
            <charges totalcharge="128.0000000" core="62.00000000" core_leakage="0.8606037277E-05" valence="66.00000000" interstitial="8.298243449" muffin-tin-total="119.7017566">
            <atom species="Pb" muffin-tin="76.65551143"/>
            <atom species="Ti" muffin-tin="18.16311654"/>
            <atom species="O" muffin-tin="8.294376196"/>
            <atom species="O" muffin-tin="8.294376196"/>
            <atom species="O" muffin-tin="8.294376196"/>
            </charges>
            <timing timetot="3.59698438644" timeinit="2.15422320366" timemat="0.251088857651" timefv="0.150968313217" timesv="0.00000000000" timerho="0.195154190063" timepot="0.845549821854" timefor="0.00000000000"/>
        </iter>
        </scl>
    </groundstate>
    </info>
    """
    output_file_path = os.path.join(tmp_path, 'info.xml')
    with open(output_file_path, "w") as xml_file:
        xml_file.write(input_string)
    assert os.path.exists(output_file_path)  # Ensure file has been written.
    results = ase.io.exciting.parse_info_out_xml(
        output_file_path, implemented_properties=['energy'])
    assert np.round(results['potential_energy']+597533.0073272572, 6) == 0.

# TODO(Fabian): Add a test for the eigenvalues.

# TODO(Fabian/dts): Add a test to make sure forces are being parsed correctly.
# I need a good exmaple file for this.