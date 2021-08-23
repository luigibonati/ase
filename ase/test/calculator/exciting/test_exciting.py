"""Test file for exciting ASE calculator."""

import os
from parameterized import parameterized
import pytest
import tempfile  # Used to create temporary directories for tests.
import unittest


import ase
from ase.build import bulk
import ase.calculators.exciting
import xml.etree.ElementTree as ET
from ase.units import Bohr, Hartree


# @pytest.mark.calculator_lite
# @pytest.mark.calculator('exciting')
# def test_exciting_bulk(factory):
#     """System level test. Ensure that at least the call doesn't fail."""
#     atoms = bulk('Si')
#     atoms.calc = factory.calc()
#     energy = atoms.get_potential_energy()
#     print(energy)


class TestExciting(unittest.TestCase):
    """Test class for all exciting unit tests."""

    def setUp(self):
        """Code to use for all tests at runtime."""
        self.test_folder_name = tempfile.mkdtemp()
        self.nitrous_oxide_atoms_obj = ase.Atoms(
            'N3O',
            positions=[
                (0, 0, 0), (1, 0, 0),
                (0, 0, 1), (0.5, 0.5, 0.5)],
            cell=[[0, 0, 1], [0, 1, 0], [1, 0, 0]],
            pbc=True)
        self.calc_dir = tempfile.mkdtemp()
        self.exciting_calc_obj = ase.calculators.exciting.Exciting(
            dir=self.calc_dir,
            kpts='3 3 3',
            species_path=self.test_folder_name,
            exciting_binary='/fshome/chm/git/exciting/bin/excitingser',
            maxscl=3)

    @parameterized.expand([[
            (3, 3, 3), '/fshome/chm/git/exciting/bin/excitingser',
            '3 3 3', '/fshome/chm/git/exciting/bin/excitingser']])
    def test_exciting_constructor(
            self, kpts, exciting_binary: str,
            expected_kpts, expected_exciting_binary: str):
        """Test write an input for exciting."""
        calc_dir = 'ase/test/calculator/exciting'
        exciting_calc = ase.calculators.exciting.Exciting(
            dir=calc_dir,
            kpts=kpts,
            species_path=self.test_folder_name,
            exciting_binary=exciting_binary,
            maxscl=3)
        # groundstate attribute ngridk returns the calculator's kpts
        self.assertEqual(
            exciting_calc.groundstate_attributes['ngridk'], expected_kpts)
        self.assertEqual(exciting_calc.dir, calc_dir)
        self.assertEqual(exciting_calc.species_path, self.test_folder_name)
        self.assertEqual(exciting_calc.exciting_binary, expected_exciting_binary)
        # Should be set to False at initialization.
        self.assertFalse(exciting_calc.converged)
        # Should be false by default unless arg is passed to constructor.
        self.assertFalse(exciting_calc.autormt)
        # Should be true by default unless arg is passed to constructor.
        self.assertTrue(exciting_calc.tshift)

    def test_add_attributes_to_element_tree(self):
        """Test adding attributes to our element tree."""
        # Default False in Exciting class constructor.
        expected_autormt = 'false'
        expected_tshift = 'true'
        expected_tforce = 'true'
        expected_maxscl = 3
        root = self.exciting_calc_obj.add_attributes_to_element_tree(
            atoms=self.nitrous_oxide_atoms_obj)
        # Let's now check that the new attributes have been added to the
        # element tree.
        self.assertEqual(
            root.find('structure').attrib['speciespath'],
            self.test_folder_name)
        self.assertEqual(
            root.find('structure').attrib['autormt'],
            expected_autormt)
        print(root.find('structure').attrib['tshift'])
        self.assertEqual(
            root.find('structure').attrib['tshift'],
            expected_tshift)
        self.assertEqual(
            root.find('groundstate').attrib['tforce'],
            expected_tforce)
        self.assertEqual(
            root.find('groundstate').attrib['maxscl'],
            str(expected_maxscl))
        self.assertEqual(
            root.find('groundstate').attrib['ngridk'],
            '3   3   3')

    def test_add_attributes_to_element_tree_with_param_dict(self):
        """Test adding attributes to element tree via param dict."""
        expected_maxscl = 3
        self.test_folder_name = tempfile.mkdtemp()
        self.nitrous_oxide_atoms_obj = ase.Atoms(
            'N3O',
            positions=[
                (0, 0, 0), (1, 0, 0),
                (0, 0, 1), (0.5, 0.5, 0.5)],
            cell=[[0, 0, 1], [0, 1, 0], [1, 0, 0]],
            pbc=True)
        param_dict = {'maxscl': '3'}
        exciting_calc_obj = ase.calculators.exciting.Exciting(
            dir=self.calc_dir,
            kpts='3 3 3',
            species_path=self.test_folder_name,
            exciting_binary='/fshome/chm/git/exciting/bin/excitingser',
            param_dict=param_dict)

        root = exciting_calc_obj.add_attributes_to_element_tree(
            atoms=self.nitrous_oxide_atoms_obj)
        self.assertEqual(
            root.find('structure').attrib['speciespath'],
            self.test_folder_name)
        # Right now the maxscl param is passed in through a param dict to the
        # exciting calculator object.
        self.assertEqual(
            root.attrib['maxscl'],
            str(expected_maxscl))

    def test_dict_to_xml(self):
        """Test adding a parameter dict to an element tree."""
        fake_xml_string = """<?xml version = "1.0"?>
        <data>
            <country_name> = "Liechtenstein">
                <rank>  1 </rank>
                </country_name>
        </data>"""
        root = ET.fromstring(fake_xml_string)
        self.assertFalse(hasattr(root, 'goose_egg'))
        fake_dict = {'goose_egg': 'true'}
        self.exciting_calc_obj.dict_to_xml(pdict=fake_dict, element=root)
        self.assertEqual(root.attrib['goose_egg'], 'true')

    def test_write(self):
        """Test write ase atoms file into an exciting input file."""
        self.exciting_calc_obj.write(atoms=self.nitrous_oxide_atoms_obj)
        self.assertTrue(os.path.isfile(self.calc_dir + '/input.xml'))
        # Now read the xml file and make sure it has the right chemical
        # symbols in the right place (e.g. under structure/species).
        tree = ET.parse(self.calc_dir + '/input.xml')
        root = tree.getroot()
        # assert the chemical symbols are N and O
        species = root.findall("structure/species")
        # We want a list of the symbols for each species we have
        species = [species[i].attrib for i in range(len(species))]
        symbols = [attribute['chemicalSymbol'] for attribute in species]
        expected_chemical_symbols = ['N', 'O']
        self.assertListEqual(symbols, expected_chemical_symbols)

    def test_read(self):
        """Test reading the exciting output info.xml file."""

        xml_string = """<?xml version="1.0"?>
            <?xml-stylesheet href="http://xml.exciting-code.org/info.xsl" type="text/xsl"?>
            <info date="2016-09-01" time="17:52:17" versionhash="" title="AFM Fe-bcc">
            <groundstate status="finished">
                <scl>
                <iter iteration="20" rms="1.6141002835615733E-008" rmslog10="-7.79206948619" deltae="8.0951858763000928E-006" deltaelog10="-5.09177317481" chgdst="2.0403852911831260E-008" chgdstlog10="-7.69028781589" fermidos="132.482182013">
                    <energies totalEnergy="-2543.25348773" fermiEnergy="0.426565746417" sum-of-eigenvalues="-1466.47893261" electronic-kinetic="2577.53462387" core-electron-kinetic="0.00000000000" Coulomb="-5011.26002189" Coulomb-potential="-3900.52860564" nuclear-nuclear="-382.039729813" electron-nuclear="-5357.91197852" Hartree="728.691686443" Madelung="-3060.99571907" xc-potential="-143.339250380" xc-effective-B-field="-0.120729244350" external-B-field="0.00000000000" exchange="-106.757424516" correlation="-2.77066519350"/>
                    <charges totalcharge="52.00000000" core="20.00000000" core_leakage="0.3814761840E-09" valence="32.00000000" interstitial="3.627763643" muffin-tin-total="48.37223636">
                    <atom species="Fe" muffin-tin="24.18611818"/>
                    <atom species="Fe" muffin-tin="24.18611818"/>
                    </charges>
                    <timing itertime="48.5351307980" timetot="1230.56626552" timeinit="0.834124858000" timemat="42.4505247870" timefv="19.9505684140" timesv="814.310320974" timerho="332.526011722" timepot="20.4947147610" timefor="0.00000000000"/>
                    <moments>
                    <momtot x="0.451215738635E-16"/>
                    <interstitial x="0.451215738635E-16"/>
                    <mommttot x="0.00000000000"/>
                    <atom species="Fe">
                        <mommt x="-1.70899579576"/>
                    </atom>
                    <atom species="Fe">
                        <mommt x="1.70899579576"/>
                    </atom>
                    </moments>
                </iter>
                <structure>
                    <crystal unitCellVolume="140.6080000" BrillouinZoneVolume="1.764125892" nktot="0" ngridk="6    6    6">
                    <basevect>5.200000000       0.000000000       0.000000000</basevect>
                    <basevect>0.000000000       5.200000000       0.000000000</basevect>
                    <basevect>0.000000000       0.000000000       5.200000000</basevect>
                    <reciprvect>1.208304867       0.000000000       0.000000000</reciprvect>
                    <reciprvect>0.000000000       1.208304867       0.000000000</reciprvect>
                    <reciprvect>0.000000000       0.000000000       1.208304867</reciprvect>
                    </crystal>
                    <species chemicalSymbol="Fe">
                    <atom x="0.00000000000" y="0.00000000000" z="0.00000000000"/>
                    <atom x="0.500010000000" y="0.500010000000" z="0.500010000000"/>
                    </species>
                </structure>
                </scl>
            </groundstate>
            </info>"""
        root = ET.fromstring(xml_string)
        pretty_string = ase.calculators.exciting.prettify(root)
        test_xml_file = os.path.join(self.calc_dir, 'info.xml')
        with open(test_xml_file, 'w') as f:
            f.write(pretty_string)
        # Now read the file, it should find the info.xml file automatically.
        self.exciting_calc_obj.read()
        expected_energy = -2543.25348773*Hartree
        self.assertAlmostEqual(self.exciting_calc_obj.energy, expected_energy)
