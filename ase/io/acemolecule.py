

import numpy as np
import ase.units
from ase.atoms import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read
from ase.data import chemical_symbols
import os

def parse_geometry(filename):
    '''Read atoms geometry from ACE-Molecule log file and put it to self.data.
    Parameters
    ==========
    filename: ACE-Molecule log file.

    Returns
    =======
    Dictionary of parsed geometry data.
    retval["Atomic_numbers"]: list of atomic numbers
    retval["Positions"]: list of [x, y, z] coordinates for each atoms.
    '''
    with open(filename, 'r') as f:
        lines = f.readlines()
        start_line = 0
        end_line = 0
        for i, line in enumerate(lines):
            if line == '====================  Atoms  =====================\n':
                start_line = i
            if start_line != 0 and len(line.split('=')) > 3:
                end_line = i
                if not start_line == end_line:
                    break
        atoms = []
        positions = []
        print(start_line, end_line)
        for i in range(start_line + 1, end_line):
            atomic_number = lines[i].split()[0]
            atoms.append(str(chemical_symbols[int(atomic_number)]))
            xyz = [float(n) for n in lines[i].split()[1:4]]
            positions.append(xyz)

        return {"Atomic_numbers": atoms, "Positions": positions}


def read_acemolecule_out(filename):
    '''Interface to ACEMoleculeReader and return values for corresponding quantity
    Parameters
    ==========
    filename: ACE-Molecule log file.
    quantity: One of atoms, energy, forces, excitation-energy.

    Returns
    =======
     - quantity = 'excitation-energy':
       returns None. This is placeholder function to run TDDFT calculations
       without IndexError.
     - quantity = 'energy':
       returns energy as float value.
     - quantity = 'forces':
       returns force of each atoms as numpy array of shape (natoms, 3).
     - quantity = 'atoms':
       returns ASE atoms object.
    '''
    data = parse_geometry(filename)
    atom_symbol = np.array(data["Atomic_numbers"])
    positions = np.array(data["Positions"])
    atoms = Atoms(atom_symbol, positions=positions)
    energy = None
    forces = None
    excitation_energy = None
#    results = {}
#    if len(results)<1:
    with open(filename, 'r') as f:
        lines = f.readlines()

    for i in range(len(lines) - 1, 1, -1):
        line = lines[i].split()
        if len(line) > 2:
            if line[0] == 'Total' and line[1] == 'energy':
                energy = float(line[3])
                break
    # energy must be modified, hartree to eV
    energy *= ase.units.Hartree

    forces = []
    for i in range(len(lines) - 1, 1, -1):
        if '!============================' in lines[i]:
            endline_num = i
        if '! Force:: List of total force in atomic unit' in lines[i]:
            startline_num = i + 2
            for j in range(startline_num, endline_num):
                forces.append(lines[j].split()[3:6])
            convert = ase.units.Hartree / ase.units.Bohr
            forces = np.array(forces, dtype=float) * convert
            break
    if not len(forces) > 0:
        forces = None

    # Set calculator to
    calc = SinglePointCalculator(atoms, energy=energy, forces=forces)
    atoms.calc = calc

    results = {}
    results['energy'] = energy
    results['atoms'] = atoms
    results['forces'] = forces
    results['excitation-energy'] = excitation_energy
    return results


def read_acemolecule_input(filename):
    '''Reads a ACE-Molecule input file
    Parameters
    ==========
    filename: ACE-Molecule input file name

    Returns
    =======
    ASE atoms object containing geometry only.
    '''
    with open(filename, 'r') as f:
        for line in f:
            if len(line.split('GeometryFilename')) > 1:
                geometryfile = line.split()[1]
                break
    atoms = read(geometryfile, format='xyz')
    return atoms

def test_acemolecule_output():
    from ase.units import Hartree, Bohr
    import pytest
    sample_outfile = """\

====================  Atoms  =====================
 1       1.000000       2.000000      -0.6
 9       -1.000000       3.000000       0.7
==================================================

Total energy       = -1.5

!================================================
! Force:: List of total force in atomic unit.
! Atom           x         y         z
! Atom   0      0.1       0.2       0.3
! Atom   1      0.5       0.6       0.7
!================================================

    """
    f = open('acemolecule_test.log','w')
    f.write(sample_outfile)
    f.close()
    #fd = StringIO(sample_outfile)
    results = read_acemolecule_out('acemolecule_test.log')
    os.system('rm acemolecule_test.log')
    atoms = results.pop('atoms')
    assert atoms.positions == pytest.approx(
        np.array([[1.0, 2.0, -0.6], [-1.0, 3.0, 0.7]]))
    assert all(atoms.symbols == 'HF')
    
    convert = ase.units.Hartree / ase.units.Bohr
    assert results.pop('forces') / convert == pytest.approx(
        np.array([[0.1, 0.2, 0.3], [0.5, 0.6, 0.7]]))
    assert results.pop('energy')/ Hartree == -1.5


def test_acemolecule_input():
    import pytest
    sample_inputfile = """\
%% BasicInformation
    Type Points
    Scaling 0.35
    Basis Sinc
    Grid Basic
    KineticMatrix Finite_Difference
    DerivativesOrder 7
    GeometryFilename acemolecule_test.xyz
    CellDimensionX 3.37316805
    CellDimensionY 3.37316805
    CellDimensionZ 3.37316805
    PointX 16
    PointY 16
    PointZ 16
    Periodicity 3
    %% Pseudopotential
        Pseudopotential 3
        PSFilePath PATH
        PSFileSuffix .PBE
    %% End
    GeometryFormat xyz
%% End
    """
    f = open('acemolecule_test.inp','w')
    f.write(sample_inputfile)
    f.close()
    atoms = Atoms(symbols='HF',positions = np.array([[1.0, 2.0, -0.6], [-1.0, 3.0, 0.7]]))
    atoms.write('acemolecule_test.xyz',format='xyz')
    atoms = read_acemolecule_input('acemolecule_test.inp')
    assert atoms.positions == pytest.approx(
        np.array([[1.0, 2.0, -0.6], [-1.0, 3.0, 0.7]]))
    assert all(atoms.symbols == 'HF')
    os.system('rm acemolecule_test.inp')
    os.system('rm acemolecule_test.xyz')

if __name__ == "__main__":
    import sys
    from ase.io import read as ACE_read
    test_acemolecule_output()
    test_acemolecule_input()
