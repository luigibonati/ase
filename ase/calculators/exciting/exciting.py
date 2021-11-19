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
# import ase.io.exciting
from ase.units import Bohr, Hartree
from ase.calculators.calculator import PropertyNotImplementedError


class Exciting:

    def update(self, atoms: ase.Atoms):
        """Initialize calc if needed then run exciting calc.

        Args:
            atoms: atom numbers, positions and
                periodic boundary conditions in an ase.Atoms obj.
        """
        # If the calculation is not over or the numbers
        # member variable hasn't been initialized we initialize
        # the atoms input.
        if (not self.converged or
                len(self.numbers) != len(atoms) or
                (self.numbers != atoms.get_atomic_numbers()).any() or
                (self.positions != atoms.get_positions()).any() or
                (self.pbc != atoms.get_pbc()).any() or
                (self.cell != atoms.get_cell()).any()):
            self.calculate(atoms)

    def initialize(self, atoms: ase.Atoms):
        """Initialize atomic information by writing input file and saving data in calculator object.

        Args:
            atoms: Holds geometry, atomic information about calculation.
        """
        # Get a list of the atomic numbers (a.m.u) save
        # it to the member variable called self.numbers.
        self.numbers = atoms.get_atomic_numbers().copy()
        # Get positions of the ASE Atoms object atom positions.
        self.positions = atoms.get_positions().copy()
        # Get the ASE Atoms object cell vectors.
        self.cell = atoms.get_cell().copy()
        # Get the periodic boundary conditions.
        self.pbc = atoms.get_pbc().copy()
        # Write ASE atoms information into input.
        self.write_input(atoms)

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

    # TODO Delete
    def calculate(self, atoms: ase.Atoms):
        """Run exciting calculation.

        Args:
            atoms: Holds geometry, atomic information about calculation.
        """
        # Write input file.
        self.initialize(atoms)

        # TODO(dts): This code portion needs a rewrite for readability.

        xmlfile = pathlib.Path(self.dir) / 'input.xml'
        # Check that the xml file exists in this location: self.dir/input.xml
        assert xmlfile.is_file()
        # Read the input and print it to the console.
        print(xmlfile.read_text())
        # Use subprocess to call the exciting binary and tell it to use input.xml as a first argument.
        argv = [self.exciting_binary, 'input.xml']
        subprocess.check_call(argv, cwd=self.dir)
        # Ensure that the calculation created output files INFO.OUT which is the
        # the text based output of exciting and info.xml which is the
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


    # TODO(Fab) Move to IO/ Use the one from inside out parsers
    def read(self):
        """ Read total energy and forces from the info.xml output file."""
        # Define where to find output file which is called
        # info.xml in exciting.
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
            # Append the total force to the forces list.
            forces.append(np.array(list(force.attrib.values())).astype(float))
        # Reshape forces so we get three columns (x,y,z) and scale units.
        self.forces = np.reshape(forces, (-1, 3)) * Hartree / Bohr
        # Check if the calculation converged.
        if str(parsed_output.find('groundstate').attrib[
                   'status']) == 'finished':
            self.converged = True
        else:
            raise RuntimeError('Calculation did not converge.')
