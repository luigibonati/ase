"""
This is the implementation of the exciting I/O functions
The functions are called with read write using the format "exciting"

"""

import numpy as np
import xml.etree.ElementTree as ET
from ase.atoms import Atoms
from ase.units import Bohr
from ase.utils import writer
from ase.calculators.exciting import atoms_to_etree
from xml.dom import minidom


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
def write_exciting(fileobj, images):
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

