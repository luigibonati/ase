"""This is the implementation of the exciting I/O functions.

The main roles these functions do is write exciting ground state
input files and read exciting ground state ouput files.

Right now these functions all written without a class to wrap them. This
could change in the future but was done to make things simpler.

These functions are primarily called by the exciting caculator in
ase/calculators/exciting/exciting.py.

See the correpsonding test file in ase/test/io/test_exciting.py.
"""

import os

import numpy as np
from typing import Dict, Union
import xml.etree.ElementTree
from xml.dom import minidom

import ase
from ase.atoms import Atoms
# from ase.calculators.exciting.exciting import ExcitingGroundStateResults
from ase.units import Bohr, Hartree


def structure_xml_to_ase_atoms(fileobj) -> ase.Atoms:
    """Reads structure from input.xml file.

    Args:
        fileobj: File handle from which data should be read.
    Returns:
        ASE atoms object with all the relevant fields filled.
    """
    # Parse file into element tree
    doc = xml.etree.ElementTree.parse(fileobj)
    root = doc.getroot()
    speciesnodes = root.find('structure').iter('species')

    symbols = []
    positions = []
    basevects = []

    # Collect data from tree
    for speciesnode in speciesnodes:
        symbol = speciesnode.get('speciesfile').split('.')[0]
        natoms = speciesnode.iter('atom')
        for atom in natoms:
            x, y, z = atom.get('coord').split()
            positions.append([float(x), float(y), float(z)])
            symbols.append(symbol)

    # scale unit cell according to scaling attributes
    if 'scale' in doc.find('structure/crystal').attrib:
        scale = float(str(doc.find('structure/crystal').attrib['scale']))
    else:
        scale = 1

    if 'stretch' in doc.find('structure/crystal').attrib:
        a, b, c = doc.find('structure/crystal').attrib['stretch'].text.split()
        stretch = np.array([float(a), float(b), float(c)])
    else:
        stretch = np.array([1.0, 1.0, 1.0])

    basevectsn = root.findall('structure/crystal/basevect')
    for basevect in basevectsn:
        x, y, z = basevect.text.split()
        basevects.append(np.array([float(x) * Bohr * stretch[0],
                                   float(y) * Bohr * stretch[1],
                                   float(z) * Bohr * stretch[2]
                                   ]) * scale)
    atoms = Atoms(symbols=symbols, cell=basevects)

    atoms.set_scaled_positions(positions)
    if 'molecule' in root.find('structure').attrib.keys():
        if root.find('structure').attrib['molecule']:
            atoms.set_pbc(False)
    else:
        atoms.set_pbc(True)

    return atoms


def prettify(elem: xml.etree.ElementTree.Element) -> str:
    """Make the XML elements prettier to read."""
    rough_string = xml.etree.ElementTree.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="\t")


def initialise_input_xml(title_text='') -> xml.etree.ElementTree.Element:
    """Initialise input.xml element tree for exciting.

    Includes a required subelements:
        * structure
        * crystal

    Args:
        title_text: Title for calculation.
    Returns:
        root: Elememnt tree root.
    """
    root = xml.etree.ElementTree.Element('input')

    root.set(
        '{http://www.w3.org/2001/XMLSchema-instance}noNamespaceSchemaLocation',
        'http://xml.exciting-code.org/excitinginput.xsd')
    title = xml.etree.ElementTree.SubElement(root, 'title')
    title.text = title_text

    structure = xml.etree.ElementTree.SubElement(root, 'structure')
    crystal = xml.etree.ElementTree.SubElement(structure, 'crystal')

    return root


def structure_element_tree(atoms: ase.Atoms) -> xml.etree.ElementTree.Element:
    """Add structure to the XML element tree.

    TODO(dts): I think we can remove this function, not sure why we need it.

    This function is basically a wrapper to call
    add_atoms_to_structure_element_tree to add geometry/chemical information
    about the system to the XML tree object.

    Args:
        atoms: ASE atoms and lattice vectors
    Returns:
        The XML element tree with structural information added.
    """
    structure = xml.etree.ElementTree.Element('crystal')
    structure = add_atoms_to_structure_element_tree(structure, atoms)
    return structure


def add_atoms_to_structure_element_tree(
        structure: xml.etree.ElementTree.Element,
        atoms: ase.Atoms) -> xml.etree.ElementTree.Element:
    """Adds lattice vectors, positions and elemental information to XML object.

    Args:
        structure: XML object that we will fill with structural information.
        atoms: Contains all the structural information we need (e.g. lattice
            vectors, element types, positions in unit cell).
    Returns:
        The XML object with the sturctural information added.
    """
    crystal = structure.find('structure/crystal')
    for vec in atoms.cell:
        base_vect = xml.etree.ElementTree.SubElement(crystal, 'basevect')
        base_vect.text = f'{vec[0] / Bohr:.14f} {vec[1] / Bohr:.14f} {vec[2] / Bohr:.14f}'

    oldsymbol = ''
    oldrmt = -1  # The old radius of the muffin tin (rmt)
    newrmt = -1
    scaled_positions = atoms.get_scaled_positions()  # positions in fractions of the unit cell
    # loop over the different elements and add corresponding species files
    for aindex, symbol in enumerate(atoms.get_chemical_symbols()):
        # TODO(speckhard): Check if rmt can be set
        if 'rmt' in atoms.arrays:
            newrmt = atoms.get_array('rmt')[aindex] / Bohr
        if symbol != oldsymbol or newrmt != oldrmt:
            speciesnode = xml.etree.ElementTree.SubElement(structure, 'species',
                                        speciesfile=f'{symbol}.xml',
                                        chemicalSymbol=symbol)
            oldsymbol = symbol
            if 'rmt' in atoms.arrays:
                oldrmt = atoms.get_array('rmt')[aindex] / Bohr
                if oldrmt > 0:
                    speciesnode.attrib['rmt'] = f'{oldrmt:.4f}'

        atom = xml.etree.ElementTree.SubElement(
            speciesnode, 'atom',
            coord=(
                f'{scaled_positions[aindex][0]:.14f} '
                f'{scaled_positions[aindex][1]:.14f} '
                f'{scaled_positions[aindex][2]:.14f}'))
        # TODO(speckhard): Can momenta be set in arrays
        if 'momenta' in atoms.arrays:
            atom.attrib['bfcmt'] = (f'{atoms.get_array("momenta")[aindex][0]:.14f} '
                                    f'{atoms.get_array("momenta")[aindex][1]:.14f} '
                                    f'{atoms.get_array("momenta")[aindex][2]:.14f}')

    return structure


def dict_to_xml(pdict: Dict, element):
    """Write dictionary k,v pairs to XML DOM object.

    The element tree argument gets modified when this function is called.

    Args:
        pdict: k,v pairs that go into the xml like file.
        element: The XML object (XML DOM object) that we want to modify
            using dictionary's k,v pairs.
    """
    for key, value in pdict.items():
        if isinstance(value, str) and key == 'text()':
            element.text = value
        elif isinstance(value, str):
            element.attrib[key] = value
        # if the value is a list, recursively call this
        # method to add each member of the list with the
        # same key for all of them.
        elif isinstance(value, list):
            for item in value:
                dict_to_xml(item, xml.etree.ElementTree.SubElement(element, key))
        # Otherwise if the value is a dictionary.
        elif isinstance(value, dict):
            if not element.findall(key):
                dict_to_xml(value, xml.etree.ElementTree.SubElement(
                    element, key))
            else:
                dict_to_xml(value, element.findall(key)[0])
        else:
            raise TypeError(f'cannot deal with key: {key}, val: {value}')


def parse_info_out_xml(full_file_path: Union[os.PathLike, str]) -> dict:
    """Read total energy and forces from the info.xml output file.

    Args:
        full_file_path: Path to the exciting calculation input.xml file.
    Returns:
        Dictionary with the outputs of the calculation.
    """
    # Probably return a dictionary as a free function.
    # Then have a light wrapper in class ExcitingGroundStateResults to call
    # parse_info_out_xml and pass the dictionary values to attributes
    # Check if calculation converged by inspecting WARNINGS.OUT

    # Try to open the output file xml.
    try:
        with open(full_file_path, 'r') as outfile:
            # Parse the XML output.
            parsed_output = xml.etree.ElementTree.parse(outfile)
    except IOError:
        raise RuntimeError(
            "Output file %s doesn't exist" % output_file)

    results = dict()
    # Find the last instance of 'totalEnergy'.
    energy = float(parsed_output.findall(
        'groundstate/scl/iter/energies')[-1].attrib[
                            'totalEnergy']) * Hartree
    results['potential_energy'] = energy

    # TODO(dts): Doesn't seem like forces are present in test exciting info.xml
    # files:
    # https://git.physik.hu-berlin.de/sol/exciting/-/blob/development/test/test_farm/groundstate/LDA_PW-PbTiO3/ref/info.xml
    # So maybe we shouldn't be parsing this all the time.
    # 
    # # Initialize forces list.
    # forces = []
    # # final all instances of 'totalforce'.
    # forcesnodes = parsed_output.findall(
    #     'groundstate/scl/structure')[-1].findall(
    #     'species/atom/forces/totalforce')
    # # Go through each force in the found instances of 'total force'.
    # for force in forcesnodes:
    #     # Append the total force to the forces list.
    #     forces.append(np.array(list(force.attrib.values())).astype(float))
    # # Reshape forces so we get three columns (x,y,z) and scale units.
    # forces = np.reshape(forces, (-1, 3)) * Hartree / Bohr
    # results['forces'] = forces

    # TODO: convergence check needed here? dts: Yes let's add this in.
    """
    # Check if the calculation converged.
    if str(parsed_output.find('groundstate').attrib[
               'status']) == 'finished':
        converged = True
    else:
        # BAD - return the status as 'finished': False
        # or converged: False
        # or it's simply not the reson
        # and the caller can decide how to handle the errors
        raise RuntimeError('Calculation did not converge.')
    """
    return results


def parse_eigval_xml(directory: Union[os.PathLike, str]) -> dict:
    """ Read eigenvalues from the eigval.xml output file.

    This function was taking from the exciting repository.

    Args:
        directory: dir to the exciting calculation.
    Rturns:
        Dictionary with the eigenvalues at different k point keys.
    """
    output_file = os.path.join(directory, 'eigval.xml')
    root = xml.etree.ElementTree.parse(output_file).getroot()
    eigval = root.attrib

    kpts = []
    for node in root.findall('kpt'):
        kpt = node.attrib
        state = []
        for subnode in node:
            state.append(subnode.attrib)
            kpt['state'] = {}  # converts list of states into a dictionary
        for item in state:
            name = item['ist']
            kpt['state'][name] = item
            kpts.append(kpt)
            eigval['kpt'] = {}
    for item in kpts:  # converts list of kpts into a dictionary
        name = item['ik']
        eigval['kpt'][name] = item

    return eigval
