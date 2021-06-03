"""ASE Calculator for the exciting DFT code."""

import os
import xml.etree.ElementTree as ET
from xml.dom import minidom
from typing import Optional, Dict

import numpy as np
import ase
# TODO(dts): use import ase for all of these imports, no need for simplifying path.
from ase.io.exciting import atoms2etree
from ase.units import Bohr, Hartree
from ase.calculators.calculator import PropertyNotImplementedError


class Exciting:
    """Class for doing exciting calculations."""
    def __init__(
                self,
                dir: str = 'calc', paramdict: Optional[Dict] = None,
                species_path: Optional[str] = None,
                exciting_binary='excitingser', kpts=(1, 1, 1),
                autormt=False, tshift=True, **kwargs):
        """Construct exciting-calculator object.

        Args:
            dir: directory in which to execute exciting.
            paramdict: Dictionary containing XML parameters. String
                values are translated to attributes, nested dictionaries are
                translated to sub elements. A list of dictionaries is
                translated to a  list of sub elements named after the key
                of which the list is the value. Default: None
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
        self.paramdict = paramdict
        # If the speciespath is not given, try to locate it.
        print(species_path)
        if species_path is None:
            try:  # TODO: check whether this dir exists.
                species_path = os.environ['EXCITINGROOT'] + '/species'
            except KeyError:
                raise RuntimeError('No species path given and no EXCITINGROOT local var found')
        try:
            assert os.isdir(species_path)
        except KeyError:
            raise RuntimeError('Species path given: %s, does not exist as a directory' % species_path)
        self.species_path = species_path
        # We initialize our _calc.s+caconverged flag indicating
        # whether the calculation is finished to False.
        self.converged = False
        self.exciting_binary = exciting_binary
        # TODO(dts): find out what this does? Add comment.
        self.autormt = autormt
        # TODO(dts): find out what this does, add comment.
        self.tshift = tshift
        # Instead of defining paramdict you can also define
        # kwargs seperately._calc.ss+ca
        self.groundstate_attributes = kwargs
        # If we can't find ngrik in kwargs and paramdict=None
        if ('ngridk' not in kwargs.keys() and not (self.paramdict)):
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
            atoms (ASE object): ASE atoms object holding
                geometry, atomic information about calculation.
        """
        # Get a list of the atomic numbers (a.m.u) save
        # it to the member variable called self.numbers.
        self.numbers = atoms.get_atomic_numbers().copy()
        # Write ASE atoms information into input.
        self.write(atoms)

    def get_potential_energy(self, atoms):
        """Get potential energy.

        Args:
            atoms (ASE Atoms Object): Positions, cell and pbc of input.

        Returns:
            Total energy of the calculation.
        """
        # Update the atoms.
        self.update(atoms)
        # Return the total energy of the calculation.
        return self.energy

    def get_forces(self, atoms):
        """Run exciting calc and get forces.

        Args:
            atoms: Positions, cell and pbc of input.

        Returns:
            forces: Total forces as a list on the structure.
        """
        # Run exciting calculation and read output.
        self.update(atoms)
        # return forces.
        return self.forces.copy()

    def get_stress(self, atoms):
        raise PropertyNotImplementedError

    def calculate(self, atoms):
        """Run exciting calculation.
        
        Args:
            atoms: Contains atomic information.
        """
        # Get positions of the ASE Atoms object atom positions.
        self.positions = atoms.get_positions().copy()
        # Get the ASE Atoms object cell vectors.
        self.cell = atoms.get_cell().copy()
        # Get the periodic boundary conditions.
        self.pbc = atoms.get_pbc().copy()
        # Write input file.
        self.initialize(atoms)

        from pathlib import Path
        xmlfile = Path(self.dir) / 'input.xml'
        assert xmlfile.is_file()
        print(xmlfile.read_text())
        argv = [self.excitingbinary, 'input.xml']
        from subprocess import check_call
        check_call(argv, cwd=self.dir)

        assert (Path(self.dir) / 'INFO.OUT').is_file()
        assert (Path(self.dir) / 'info.xml').exists()

        # Command line call to calculate exciting.
        # syscall = 'cd ' + self.dir + 's;' + self.exciting_binary + 's;'
        # print(syscall)
        # Ensure that the syscall doesn't return an error.
        # assert os.system(syscall) == 0
        # Read the results of the calculation.
        self.read()

    def write(self, atoms):
        """Write atomic info inputs to an xml."""
        # Check if the directory where we want to save our file exists.
        # If not, create the directory.
        if not os.path.isdir(self.dir):
            os.mkdir(self.dir)
        # Create an XML Document Object Model (DOM) where we can
        # then assign different attributes of the DOM.
        root = atoms2etree(atoms)
        # Assign the species path to the XML DOM.
        root.find('structure').attrib['speciespath'] = self.species_path
        # Assign the autormt boolean.
        root.find('structure').attrib['autormt'] = str(
            self.autormt).lower()
        # Assign the tshift bool. If true crystal is shifted so
        # closest atom to origin is now at origin.
        root.find('structure').attrib['tshift'] = str(
            self.tshift).lower()

        def prettify(elem):
            "Make the element prettier to read."
            rough_string = ET.tostring(elem, 'utf-8')
            reparsed = minidom.parseString(rough_string)
            return reparsed.toprettyxml(indent="\t")

        # Check if paramdict is not None.
        if self.paramdict:
            # Assign dict key values to XML dom object root.
            self.dicttoxml(self.paramdict, root)
            with open(self.dir + '/input.xml', 'w') as fd:
                fd.write(prettify(root))
        else:  # For undefined paramdict.
            groundstate = ET.SubElement(root, 'groundstate', tforce='true')
            for key, value in self.groundstate_attributes.items():
                if key == 'title':
                    root.findall('title')[0].text = value
                else:
                    groundstate.attrib[key] = str(value)
            with open(self.dir + '/input.xml', 'w') as fd:
                fd.write(prettify(root))
                fd.close()

    def dicttoxml(self, pdict: Dict, element):
        """Write dictionary k,v paris to XML DOM object.

        Args:

            pdict: Dictionary with k,v pairs that go into the xml like file.
            element: The XML object (XML DOM object) that we want to modify using dict k,v pairs.
        """
        for key, value in pdict.items():
            if (isinstance(value, str) and key == 'text()'):
                element.text = value
            elif (isinstance(value, str)):
                element.attrib[key] = value
            elif (isinstance(value, list)):
                # if the value is a list, recursively call this
                # method to add each member of the list with the
                # same key for all of them.
                for item in value:
                    self.dicttoxml(item, ET.SubElement(element, key))
            # TODO: understand what this does.
            elif (isinstance(value, dict)):
                if(element.findall(key) == []):
                    self.dicttoxml(value, ET.SubElement(element, key))
                else:
                    self.dicttoxml(value, element.findall(key)[0])
            else:
                print('cannot deal with', key, '=', value)

    def read(self):
        """ Read total energy and forces from info.xml output."""
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
        info = ET.parse(outfile)
        # Find the last istance of 'totalEnergy'.
        self.energy = float(parsed_output.findall(
            'groundstate/scl/iter/energies')[-1].attrib[
                'totalEnergy']) * Hartree
        # Initialize forces list.
        forces = []
        # final all instances of 'totalforce'.
        forcesnodes = info.findall(
                'groundstate/scl/structure')[-1].findall(
                    'species/atom/forces/totalforce')
        # Go through each force in the found instances of 'total force'.
        for force in forcesnodes:
            # Apend the total force to the forces list.
            forces.append(np.array(list(force.attrib.values())).astype(float))
        # Reshape forces so we get three columns (x,y,z) and scale units.
        self.forces = np.reshape(forces, (-1, 3)) * Hartree / Bohr
        # Check if the calculation converged.
        if str(info.find('groundstate').attrib['status']) == 'finished':
            self.converged = True
        else:
            raise RuntimeError('Calculation did not converge.')
