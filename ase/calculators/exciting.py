"""ASE Calculator for the exciting DFT code."""

import os
import xml.etree.ElementTree as ET
from xml.dom import minidom
from typing import Optional, Dict
import pathlib
import subprocess

import numpy as np
import ase
# TODO(dts): use import ase for all of these imports, no need for simplifying path.
import ase.io.exciting
from ase.units import Bohr, Hartree
from ase.calculators.calculator import PropertyNotImplementedError


class Exciting:
    """Class for doing exciting calculations.

    You can find lots of information regarding the XML schema for exciting
    in the appendix of this thesis:
    https://pure.unileoben.ac.at/portal/files/1842879/AC08976214n01vt.pdf
    """

    def __init__(
                self,
                dir: str = 'calc', param_dict: Optional[Dict] = None,
                species_path: Optional[str] = None,
                exciting_binary='excitingser', kpts=(1, 1, 1),
                autormt=False, tshift=True, **kwargs):
        """Construct exciting-calculator object.

        Args:
            dir: directory in which to execute exciting.
            param_dict: Dictionary containing XML parameters. String
                values are translated to attributes, nested dictionaries are
                translated to sub elements. A list of dictionaries is
                translated to a  list of sub elements named after the key
                of which the list is the value. The keys of the dict should
                be a list of strings not floats.
            species_path: string
                Directory or URL to look up species files folder.
            exciting_binary: Path to executable of exciting code.
                Default: 'excitingser'
            kpts: Number of k-points. List len should be 3.
            autormt: Assign the autormt boolean. If true the muffin tin
                radius is set automatically by rmtapm.
            kwargs: List of key, value pairs to be
                converted into groundstate attributes.
        """
        # Assign member variables using contructor arguments.
        self.dir = dir
        self.energy = None
        self.param_dict = param_dict
        # If the speciespath is not given, try to locate it.
        if species_path is None:
            try:  # TODO: check whether this dir exists.
                species_path = os.environ['EXCITINGROOT'] + '/species'
            except KeyError:
                raise RuntimeError(
                    'No species path given and no EXCITINGROOT '
                    'local var found')
        else:  # Try to see if the species path directory actually exists.
            try:
                assert os.path.isdir(species_path)
            except KeyError:
                raise RuntimeError(
                    'Species path given: %s, '
                    'does not exist as a directory' % species_path)
        self.species_path = species_path
        # We initialize our _calc.s+caconverged flag indicating
        # whether the calculation is finished to False.
        self.converged = False
        self.exciting_binary = exciting_binary
        # If true, the radius of the muffin tin is set automatically.
        self.autormt = autormt
        # If tshift is "true", the crystal is shifted such that the atom
        # closest to the origin is exactly at the origin.
        self.tshift = tshift
        # Instead of defining param_dict you can also define
        # kwargs seperately._calc.ss+ca
        self.groundstate_attributes = kwargs
        # If we can't find ngrik in kwargs and param_dict=None
        if ('ngridk' not in kwargs.keys() and not (self.param_dict)):
            # Set the groundstate attributes ngridk value
            # using the kpts constructure param. The join and map
            # convert [2, 2, 2] into '2 2 2'.
            self.groundstate_attributes[
                'ngridk'] = ' '.join(map(str, kpts))

    def update(self, atoms: ase.Atoms):
        """Initialize calc if needed then run exciting calc.

        Args:
            atoms: atom numbers, positions and
                periodic boundary conditions in an ase.Atoms obj.
        """
        # If the calculation is not over or the numbers
        # member variable hasn't been initalized we initalize
        # the atoms input.
        if (
                not self.converged or
                len(self.numbers) != len(atoms) or
                (self.numbers != atoms.get_atomic_numbers()).any()):
            self.initialize(atoms)
            self.calculate(atoms)
        # Otherwise run the calculations with this ASE atoms object.
        elif ((self.positions != atoms.get_positions()).any() or
              (self.pbc != atoms.get_pbc()).any() or
              (self.cell != atoms.get_cell()).any()):
            self.calculate(atoms)

    def initialize(self, atoms: ase.Atoms):
        """Initialize atomic information by writing input file.

        Args:
            atoms: Holds geometry, atomic information about calculation.
        """
        # Get a list of the atomic numbers (a.m.u) save
        # it to the member variable called self.numbers.
        self.numbers = atoms.get_atomic_numbers().copy()
        # Write ASE atoms information into input.
        self.write(atoms)

    def get_potential_energy(self, atoms: ase.Atoms):
        """Get potential energy.

        Args:
            atoms: Holds geometry, atomic information about calculation.

        Returns:
            Total energy of the calculation.
        """
        # Update the atoms.
        self.update(atoms)
        # Return the total energy of the calculation.
        return self.energy

    def get_forces(self, atoms: ase.Atoms):
        """Run exciting calculation and get forces.

        Args:
            atoms: Holds geometry, atomic information about calculation.

        Returns:
            forces: Total forces as a list on the structure.
        """
        # Run exciting calculation and read output.
        self.update(atoms)
        # return forces.
        return self.forces.copy()

    def get_stress(self, atoms: ase.Atoms):
        """Get the stress on the unit cell.

        This method has not been implemented.
        TODO(dts): Talk to Alex and see if this has been implemented in exciting.

        Args:
            atoms: Holds geometry, atomic information about calculation.

        Raises:
            PropertyNotImplementedError every call since this function needs to be implemented.
        """
        raise PropertyNotImplementedError

    def calculate(self, atoms: ase.Atoms):
        """Run exciting calculation.

        Args:
            atoms: Holds geometry, atomic information about calculation.
        """
        # Get positions of the ASE Atoms object atom positions.
        self.positions = atoms.get_positions().copy()
        # Get the ASE Atoms object cell vectors.
        self.cell = atoms.get_cell().copy()
        # Get the periodic boundary conditions.
        self.pbc = atoms.get_pbc().copy()
        # Write input file.
        self.initialize(atoms)

        #TODO(dts): This code portion needs a rewrite for readability.

        xmlfile = pathlib.Path(self.dir) / 'input.xml'
        # CHeck that the xml file exists in this location: self.dir/input.xml 
        assert xmlfile.is_file()
        # Read the input and print it to the console.
        print(xmlfile.read_text())
        # Use subprocess to call the exciting binary and tell it to use input.xml as a first argument.
        argv = [self.excitingbinary, 'input.xml']
        subprocess.check_call(argv, cwd=self.dir)
        # Ensure that the calculation created output files INFO.OUT which is the
        # the text based output of excicing and info.xml which is the 
        assert (pathlib.Path(self.dir) / 'INFO.OUT').is_file()
        assert (pathlib.Path(self.dir) / 'info.xml').exists()

        # This was the previous code portion to call exciting.

        # Command line call to calculate exciting.
        # syscall = 'cd ' + self.dir + 's;' + self.exciting_binary + 's;'
        # print(syscall)
        # Ensure that the syscall doesn't return an error.
        # assert os.system(syscall) == 0
        # Read the results of the calculation.
        self.read()

    def add_attributes_to_element_tree(self, atoms: ase.Atoms):
        """Adds attributes to the element tree.

        The element tree created with ase.io.exciting.atoms_to_tree
        is missing a few attributes that are specified in the __init__()
        method of this class. We add them to our element tree.

        Args:
            atoms: Holds geometry and atomic information of the unit cell.

        Returns:
            An xml element tree.
        """
        # Create an XML Document Object Model (DOM) where we can
        # then assign different attributes of the DOM. `root` holds the root
        # of the element tree that is populated with basis vectors, chemical
        # symbols and the like.
        root = ase.io.exciting.atoms_to_etree(atoms)
        # We have to add a few more attributes to the element tree before
        # writing the xml input file. Assign the species path.
        root.find('structure').attrib['speciespath'] = self.species_path
        # Assign the autormt boolean.
        root.find('structure').attrib['autormt'] = str(
            self.autormt).lower()
        # Assign the tshift bool. If true crystal is shifted so
        # closest atom to origin is now at origin.
        root.find('structure').attrib['tshift'] = str(
            self.tshift).lower()
        # Check if param_dict is not None.
        if self.param_dict:
            # Assign dict key values to XML dom object root.
            self.dict_to_xml(self.param_dict, root)
        else:  # For an undefined param_dict.
            groundstate = ET.SubElement(root, 'groundstate', tforce='true')
            for key, value in self.groundstate_attributes.items():
                if key == 'title':
                    root.findall('title')[0].text = value
                else:
                    groundstate.attrib[key] = str(value)
        return root

    def write(self, atoms: ase.Atoms):
        """Write atomic info inputs to an xml.

        Write input parameters into an xml file that will be used by the
        exciting binary to run a calculation.

        Args:
            atoms: Holds geometry and atomic information of the unit cell.
        """
        # Check if the directory where we want to save our file exists.
        # If not, create the directory.
        if not os.path.isdir(self.dir):
            os.mkdir(self.dir)
        root = self.add_attributes_to_element_tree(atoms=atoms)
        with open(os.path.join(self.dir, 'input.xml'), 'w') as fd:
            # Prettify makes the output alot nicer to read.
            fd.write(prettify(root))

    def dict_to_xml(self, pdict: Dict, element):
        """Write dictionary k,v paris to XML DOM object.

        Args:
            pdict: k,v pairs that go into the xml like file.
            element: The XML object (XML DOM object) that we want to modify
                using dictionary's k,v pairs.
        """
        for key, value in pdict.items():
            if (isinstance(value, str) and key == 'text()'):
                element.text = value
            elif (isinstance(value, str)):
                element.attrib[key] = value
            # if the value is a list, recursively call this
            # method to add each member of the list with the
            # same key for all of them.
            elif (isinstance(value, list)):
                for item in value:
                    self.dict_to_xml(item, ET.SubElement(element, key))
            # Otherwise if the value is a dictionary.
            elif (isinstance(value, dict)):
                if(element.findall(key) == []):
                    self.dict_to_xml(value, ET.SubElement(element, key))
                else:
                    self.dict_to_xml(value, element.findall(key)[0])
            else:
                raise TypeError(f'cannot deal with key: {key}, val: {value}')

    def read(self):
        """ Read total energy and forces from the info.xml output file."""
        # Define where to find output file which is called
        # info.xml in exciing.
        output_file = self.dir + '/info.xml'
        # Try to open the output file.
        try:
            with open(output_file, 'r') as outfile:
                # Parse the XML output.
                parsed_output = ET.parse(outfile)
        except IOError:
            raise RuntimeError(
                "Output file %s doesn't exist" % output_file)
        # Find the last istance of 'totalEnergy'.
        self.energy = float(parsed_output.findall(
            'groundstate/scl/iter/energies')[-1].attrib[
                'totalEnergy']) * Hartree
        # Initialize forces list.
        forces = []
        # final all instances of 'totalforce'.
        forcesnodes = parsed_output.findall(
                'groundstate/scl/structure')[-1].findall(
                    'species/atom/forces/totalforce')
        # Go through each force in the found instances of 'total force'.
        for force in forcesnodes:
            # Apend the total force to the forces list.
            forces.append(np.array(list(force.attrib.values())).astype(float))
        # Reshape forces so we get three columns (x,y,z) and scale units.
        self.forces = np.reshape(forces, (-1, 3)) * Hartree / Bohr
        # Check if the calculation converged.
        if str(parsed_output.find('groundstate').attrib[
                'status']) == 'finished':
            self.converged = True
        else:
            raise RuntimeError('Calculation did not converge.')


def prettify(elem):
    """Make the XML elements prettier to read."""
    rough_string = ET.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="\t")
