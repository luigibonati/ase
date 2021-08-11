"""Test file for exciting ASE calculator."""
from parameterized import parameterized
from collections.abc import Sequence
from typing import Tuple
import tempfile  # Used to create temporary directories for tests.
import xml.etree.ElementTree as ET
from xml.dom import minidom
from ase.build import bulk
import ase.calculators.exciting as exciting
import pytest
import os
import shutil

# @pytest.mark.calculator_lite
# @pytest.mark.calculator('exciting')
# def test_exciting_bulk(factory):
#     """System level test. Ensure that at least the call doesn't fail."""
#     atoms = bulk('Si')
#     atoms.calc = factory.calc()
#     energy = atoms.get_potential_energy()
#     print(energy)


class TestExciting:
    """Test class for all exciting unit tests."""

    @pytest.fixture(autouse=True)
    def set_up(self):
        """Code to use for all tests at runtime."""
        self.test_folder_name = tempfile.mkdtemp()
        yield
        shutil.rmtree(self.test_folder_name)

    @pytest.fixture
    def calculator(self):
        return exciting.Exciting(species_path=self.test_folder_name)

    @parameterized.expand([[
        (3, 3, 3), '/fshome/chm/git/exciting/bin/excitingser',
        '3 3 3', '/fshome/chm/git/exciting/bin/excitingser'],
        [(1, 2, 3), '/foo/bar',
         '1 2 3', '/foo/bar']
    ])
    def test_exciting_constructor(self, kpts: Tuple[int], exciting_binary: str, expected_kpts: str,
                                  expected_exciting_binary: str):
        """Test write an input for exciting."""
        calc_dir = 'ase/test/calculator/exciting'
        exciting_calc = exciting.Exciting(
            dir=calc_dir,
            kpts=kpts,
            species_path=self.test_folder_name,
            exciting_binary=exciting_binary,
            maxscl=3)
        # groundstate attribute ngridk returns the calculator's kpts
        assert exciting_calc.groundstate_attributes['ngridk'] == expected_kpts
        assert exciting_calc.dir == calc_dir
        assert exciting_calc.species_path == self.test_folder_name
        assert exciting_calc.exciting_binary == expected_exciting_binary
        # Should be set to False at initialization.
        assert not exciting_calc.converged
        # Should be false by default unless arg is passed to constructor.
        assert not exciting_calc.autormt
        # Should be true by default unless arg is passed to constructor.
        assert exciting_calc.tshift

    def test_exciting_constructor_2(self):
        with pytest.raises(RuntimeError, match='No species path given and no EXCITINGROOT '
                    'local var found'):
            exciting.Exciting()

    def test_exciting_constructor_3(self):
        os.mkdir(self.test_folder_name + '/species')
        os.environ['EXCITINGROOT'] = self.test_folder_name
        calc = exciting.Exciting()
        assert calc.species_path == self.test_folder_name + '/species'

    def test_exciting_constructor_4(self):
        with pytest.raises(RuntimeError, match='Species path given'):
            calc = exciting.Exciting(species_path=self.test_folder_name + '/species')

    def test_exciting_constructor_5(self):
        calc = exciting.Exciting(ngridk='1 2 3')
        assert calc.groundstate_attributes['ngridk'] == '1 2 3'

    def test_write(self):
        """Test the write method"""
        pass

    def test_dicttoxml_1(self, calculator):
        element = ET.Element('root')
        dictionary = {'text()': 'test'}
        calculator.dicttoxml(dictionary, element)
        assert element.text.__eq__('test')

    def test_dicttoxml_2(self, calculator):
        element = ET.Element('root')
        dictionary = {'number': '2'}
        calculator.dicttoxml(dictionary, element)
        assert element.attrib.get('number') == '2'

    def test_dicttoxml_3(self, calculator):
        element = ET.Element('root')
        dictionary = {'sub': [{'text()': 'test', 'number': '2'}]}
        calculator.dicttoxml(dictionary, element)
        assert isinstance(element.find('./sub'), ET.Element)
        assert element.find('./sub').text == 'test'
        assert element.find('./sub').attrib.get('number') == '2'

    def test_dicttoxml_4(self, calculator):
        element = ET.Element('root')
        dictionary = {'sub': {'number': '2'}}
        calculator.dicttoxml(dictionary, element)
        assert isinstance(element.find('./sub'), ET.Element)
        assert element.find('./sub').attrib.get('number') == '2'

    def test_dicttoxml_4(self, calculator):
        element = ET.Element('root')
        ET.SubElement(element, 'sub')
        ET.SubElement(element, 'sub')
        dictionary = {'sub': {'number': '2'}}
        calculator.dicttoxml(dictionary, element)
        sub_elements = element.findall('./sub')
        assert len(sub_elements) == 2
        assert sub_elements[0].attrib.get('number') == '2'
        assert len(sub_elements[1].keys()) == 0

    def test_dicttoxml_5(self, calculator, capsys):
        element = ET.Element('root')
        dictionary = {'sub': 1}
        calculator.dicttoxml(dictionary, element)
        captured = capsys.readouterr()
        print(captured.out)
        assert captured.out.startswith('cannot deal with sub = 1')
