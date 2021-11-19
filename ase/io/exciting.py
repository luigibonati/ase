"""
This is the implementation of the exciting I/O functions
The functions are called with read write using the format "exciting"

"""

import numpy as np
import xml.etree.ElementTree as ET
import ase
from ase.atoms import Atoms
from ase.units import Bohr
from ase.utils import writer
from xml.dom import minidom
from typing import Optional, Dict


def read_exciting(fileobj, index=-1):
    """Reads structure from exiting xml file.

    Parameters
    ----------
    fileobj: file object
        File handle from which data should be read.

    Other parameters
    ----------------
    index: integer -1
        Not used in this implementation.
    """

    # Parse file into element tree
    doc = ET.parse(fileobj)
    root = doc.getroot()
    speciesnodes = root.find('structure').iter('species')
    symbols = []
    positions = []
    basevects = []
    atoms = None
    # Collect data from tree
    for speciesnode in speciesnodes:
        symbol = speciesnode.get('speciesfile').split('.')[0]
        natoms = speciesnode.iter('atom')
        for atom in natoms:
            x, y, z = atom.get('coord').split()
            positions.append([float(x), float(y), float(z)])
            symbols.append(symbol)
    # scale unit cell accorting to scaling attributes
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


@writer
def write_exciting_old(fileobj, images):
    """writes exciting input structure in XML

    Parameters
    ----------
    filename : str
        Name of file to which data should be written.
    images : Atom Object or List of Atoms objects
        This function will write the first Atoms object to file.

    Returns
    -------
    """
    root = atoms_to_etree(images)
    # The rest of this is the prettify function.
    rough_string = ET.tostring(root, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    pretty = reparsed.toprettyxml(indent="\t")
    fileobj.write(pretty)


@writer
def write_exciting(fileobj, images, autormt, species_path, tshift, param_dict, groundstate_attributes):
    """writes exciting input structure in XML

    Parameters
    ----------
    fileobj : File Object
        File Object returned by call to open.
    images : ASE Atoms Object or List of Atoms objects
        This function will write the first Atoms object to file.
    -------
    """
    # Check if the directory where we want to save our file exists.
    # If not, create the directory.
    root = add_attributes_to_element_tree(images, autormt, species_path, tshift, param_dict)
    # Prettify makes the output a lot nicer to read.
    fileobj.write(prettify(root))


def add_attributes_to_element_tree(atoms: ase.Atoms, autormt, species_path, tshift, param_dict):
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
    root = atoms_to_etree(atoms)
    # We have to add a few more attributes to the element tree before
    # writing the xml input file. Assign the species path.
    root.find('structure').attrib['speciespath'] = species_path
    # Assign the autormt boolean.
    root.find('structure').attrib['autormt'] = str(
        autormt).lower()
    # Assign the tshift bool. If true crystal is shifted so
    # closest atom to origin is now at origin.
    root.find('structure').attrib['tshift'] = str(
        tshift).lower()
    # Assign dict key values to XML dom object root.
    if param_dict:
        dict_to_xml(param_dict, root)
    else:
        ET.SubElement(root, 'groundstate', tforce='true')
    return root


def prettify(elem):
    """Make the XML elements prettier to read."""
    rough_string = ET.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="\t")


def dict_to_xml(pdict: Dict, element):
    """Write dictionary k,v pairs to XML DOM object.

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
                dict_to_xml(item, ET.SubElement(element, key))
        # Otherwise if the value is a dictionary.
        elif isinstance(value, dict):
            if not element.findall(key):
                dict_to_xml(value, ET.SubElement(element, key))
            else:
                dict_to_xml(value, element.findall(key)[0])
        else:
            raise TypeError(f'cannot deal with key: {key}, val: {value}')


def atoms_to_etree(ase_atoms_obj) -> ET.Element:
    """This function creates the XML DOM corresponding
     to the structure for use in write and calculator

    Parameters
    ----------

    ase_atoms_obj : Atom Object or List of Atoms objects

    Returns
    -------
    root : etree object
        Element tree of exciting input file containing the structure
    """
    # checks whether a list of ase atom objects is passed or a single ase atoms object
    if not isinstance(ase_atoms_obj, (list, tuple)):
        ase_atoms_obj_list = [ase_atoms_obj]
    else:
        ase_atoms_obj_list = ase_atoms_obj
    root = ET.Element('input')
    root.set(
        '{http://www.w3.org/2001/XMLSchema-instance}noNamespaceSchemaLocation',
        'http://xml.exciting-code.org/excitinginput.xsd')

    title = ET.SubElement(root, 'title')
    title.text = ''
    structure = ET.SubElement(root, 'structure')
    crystal = ET.SubElement(structure, 'crystal')
    # if a list of ase atoms is passed use the first one
    atoms = ase_atoms_obj_list[0]
    for vec in atoms.cell:
        basevect = ET.SubElement(crystal, 'basevect')
        # use f string here and fix this.
        basevect.text = f'{vec[0] / Bohr:.14f} {vec[1] / Bohr:.14f} {vec[2] / Bohr:.14f}'

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
            speciesnode = ET.SubElement(structure, 'species',
                                        speciesfile=f'{symbol}.xml',
                                        chemicalSymbol=symbol)
            oldsymbol = symbol
            if 'rmt' in atoms.arrays:
                oldrmt = atoms.get_array('rmt')[aindex] / Bohr
                if oldrmt > 0:
                    speciesnode.attrib['rmt'] = f'{oldrmt:.4f}'

        atom = ET.SubElement(speciesnode, 'atom', coord=(f'{scaled_positions[aindex][0]:.14f} '
                                                         f'{scaled_positions[aindex][1]:.14f} '
                                                         f'{scaled_positions[aindex][2]:.14f}'))
        # TODO(speckhard): Can momenta be set in arrays
        if 'momenta' in atoms.arrays:
            atom.attrib['bfcmt'] = (f'{atoms.get_array("momenta")[aindex][0]:.14f} '
                                    f'{atoms.get_array("momenta")[aindex][1]:.14f} '
                                    f'{atoms.get_array("momenta")[aindex][2]:.14f}')

    return root
