"""
This module contains functionality for reading and writing an ASE
Atoms object in VASP POSCAR format.

"""

import os
import re

import ase.units
from ase.utils import basestring, reader, writer
from ase.io.formats import ImageIterator


__all__ = ['read_vasp', 'read_vasp_out', 'iread_vasp_out',
           'read_vasp_xdatcar', 'read_vasp_xml',
           'write_vasp']


def get_atomtypes(fname):
    """Given a file name, get the atomic symbols.

    The function can get this information from OUTCAR and POTCAR
    format files.  The files can also be compressed with gzip or
    bzip2.

    """
    atomtypes = []
    if fname.find('.gz') != -1:
        import gzip
        f = gzip.open(fname)
    elif fname.find('.bz2') != -1:
        import bz2
        f = bz2.BZ2File(fname)
    else:
        f = open(fname)
    for line in f:
        if line.find('TITEL') != -1:
            atomtypes.append(line.split()[3].split('_')[0].split('.')[0])
    return atomtypes


def atomtypes_outpot(posfname, numsyms):
    """Try to retrieve chemical symbols from OUTCAR or POTCAR

    If getting atomtypes from the first line in POSCAR/CONTCAR fails, it might
    be possible to find the data in OUTCAR or POTCAR, if these files exist.

    posfname -- The filename of the POSCAR/CONTCAR file we're trying to read

    numsyms -- The number of symbols we must find

    """
    import os.path as op
    import glob

    # First check files with exactly same name except POTCAR/OUTCAR instead
    # of POSCAR/CONTCAR.
    fnames = [posfname.replace('POSCAR', 'POTCAR').replace('CONTCAR',
                                                           'POTCAR')]
    fnames.append(posfname.replace('POSCAR', 'OUTCAR').replace('CONTCAR',
                                                               'OUTCAR'))
    # Try the same but with compressed files
    fsc = []
    for fn in fnames:
        fsc.append(fn + '.gz')
        fsc.append(fn + '.bz2')
    for f in fsc:
        fnames.append(f)
    # Finally try anything with POTCAR or OUTCAR in the name
    vaspdir = op.dirname(posfname)
    fs = glob.glob(vaspdir + '*POTCAR*')
    for f in fs:
        fnames.append(f)
    fs = glob.glob(vaspdir + '*OUTCAR*')
    for f in fs:
        fnames.append(f)

    tried = []
    files_in_dir = os.listdir('.')
    for fn in fnames:
        if fn in files_in_dir:
            tried.append(fn)
            at = get_atomtypes(fn)
            if len(at) == numsyms:
                return at

    raise IOError('Could not determine chemical symbols. Tried files ' +
                  str(tried))


def get_atomtypes_from_formula(formula):
    """Return atom types from chemical formula (optionally prepended
    with and underscore).
    """
    from ase.symbols import string2symbols
    symbols = string2symbols(formula.split('_')[0])
    atomtypes = [symbols[0]]
    for s in symbols[1:]:
        if s != atomtypes[-1]:
            atomtypes.append(s)
    return atomtypes


@reader
def read_vasp(filename='CONTCAR'):
    """Import POSCAR/CONTCAR type file.

    Reads unitcell, atom positions and constraints from the POSCAR/CONTCAR
    file and tries to read atom types from POSCAR/CONTCAR header, if this fails
    the atom types are read from OUTCAR or POTCAR file.
    """

    from ase import Atoms
    from ase.constraints import FixAtoms, FixScaled
    from ase.data import chemical_symbols
    import numpy as np

    f = filename
    # The first line is in principle a comment line, however in VASP
    # 4.x a common convention is to have it contain the atom symbols,
    # eg. "Ag Ge" in the same order as later in the file (and POTCAR
    # for the full vasp run). In the VASP 5.x format this information
    # is found on the fifth line. Thus we save the first line and use
    # it in case we later detect that we're reading a VASP 4.x format
    # file.
    line1 = f.readline()

    lattice_constant = float(f.readline().split()[0])

    # Now the lattice vectors
    a = []
    for ii in range(3):
        s = f.readline().split()
        floatvect = float(s[0]), float(s[1]), float(s[2])
        a.append(floatvect)

    basis_vectors = np.array(a) * lattice_constant

    # Number of atoms. Again this must be in the same order as
    # in the first line
    # or in the POTCAR or OUTCAR file
    atom_symbols = []
    numofatoms = f.readline().split()
    # Check whether we have a VASP 4.x or 5.x format file. If the
    # format is 5.x, use the fifth line to provide information about
    # the atomic symbols.
    vasp5 = False
    try:
        int(numofatoms[0])
    except ValueError:
        vasp5 = True
        atomtypes = numofatoms
        numofatoms = f.readline().split()

    # check for comments in numofatoms line and get rid of them if necessary
    commentcheck = np.array(['!' in s for s in numofatoms])
    if commentcheck.any():
        # only keep the elements up to the first including a '!':
        numofatoms = numofatoms[:np.arange(len(numofatoms))[commentcheck][0]]

    if not vasp5:
        atomtypes = line1.split()

        numsyms = len(numofatoms)
        if len(atomtypes) < numsyms:
            # First line in POSCAR/CONTCAR didn't contain enough symbols.

            # Sometimes the first line in POSCAR/CONTCAR is of the form
            # "CoP3_In-3.pos". Check for this case and extract atom types
            if len(atomtypes) == 1 and '_' in atomtypes[0]:
                atomtypes = get_atomtypes_from_formula(atomtypes[0])
            else:
                atomtypes = atomtypes_outpot(f.name, numsyms)
        else:
            try:
                for atype in atomtypes[:numsyms]:
                    if atype not in chemical_symbols:
                        raise KeyError
            except KeyError:
                atomtypes = atomtypes_outpot(f.name, numsyms)

    for i, num in enumerate(numofatoms):
        numofatoms[i] = int(num)
        [atom_symbols.append(atomtypes[i]) for na in range(numofatoms[i])]

    # Check if Selective dynamics is switched on
    sdyn = f.readline()
    selective_dynamics = sdyn[0].lower() == 's'

    # Check if atom coordinates are cartesian or direct
    if selective_dynamics:
        ac_type = f.readline()
    else:
        ac_type = sdyn
    cartesian = ac_type[0].lower() == 'c' or ac_type[0].lower() == 'k'
    tot_natoms = sum(numofatoms)
    atoms_pos = np.empty((tot_natoms, 3))
    if selective_dynamics:
        selective_flags = np.empty((tot_natoms, 3), dtype=bool)
    for atom in range(tot_natoms):
        ac = f.readline().split()
        atoms_pos[atom] = (float(ac[0]), float(ac[1]), float(ac[2]))
        if selective_dynamics:
            curflag = []
            for flag in ac[3:6]:
                curflag.append(flag == 'F')
            selective_flags[atom] = curflag
    if cartesian:
        atoms_pos *= lattice_constant
    atoms = Atoms(symbols=atom_symbols, cell=basis_vectors, pbc=True)
    if cartesian:
        atoms.set_positions(atoms_pos)
    else:
        atoms.set_scaled_positions(atoms_pos)
    if selective_dynamics:
        constraints = []
        indices = []
        for ind, sflags in enumerate(selective_flags):
            if sflags.any() and not sflags.all():
                constraints.append(FixScaled(atoms.get_cell(), ind, sflags))
            elif sflags.all():
                indices.append(ind)
        if indices:
            constraints.append(FixAtoms(indices))
        if constraints:
            atoms.set_constraint(constraints)
    return atoms


class OUTCARChunck:
    def __init__(self, lines, natoms, symbols, constraints):
        self.lines = lines
        self.natoms = natoms
        self.symbols = symbols
        self.constraints = constraints

    def build(self):
        return _read_outcar_frame(self.lines, self.natoms,
                                  self.symbols, self.constraints)


def _read_outcar_frame(lines, natoms, symbols, constraints):
    import numpy as np
    from ase.calculators.singlepoint import SinglePointCalculator
    from ase import Atoms

    magnetization = []
    magmom = None
    stress = None
    atoms = Atoms(symbols=symbols, pbc=True, constraint=constraints)

    cl = _cl                    # Aliasing

    forces = np.zeros((natoms, 3))
    positions = np.zeros((natoms, 3))
    # Parse each atoms object
    for n, line in enumerate(lines):
        line = line.strip()
        if 'direct lattice vectors' in line:
            cell = []
            for i in range(3):
                parts = cl(lines[n + i + 1]).split()
                cell += [list(map(float, parts[0:3]))]
            atoms.set_cell(cell)
        elif 'magnetization (x)' in line:
            nskip = 4           # Skip some lines
            magnetization = [float(cl(lines[n + i + nskip]).split()[4])
                             for i in range(natoms)]
        elif 'number of electron' in line:
            parts = cl(line).split()
            if len(parts) > 5 and parts[0].strip() != "NELECT":
                magmom = float(parts[5])
        elif 'in kB ' in line:
            stress = -np.asarray([float(a) for a in cl(line).split()[2:]])
            stress = stress[[0, 1, 2, 4, 5, 3]] * 1e-1 * ase.units.GPa
        elif 'POSITION          ' in line:
            nskip = 2
            for i in range(natoms):
                parts = list(map(float, cl(lines[n + i + nskip]).split()))
                positions[i] = parts[0:3]
                forces[i] = parts[3:6]
            atoms.set_positions(positions)
        elif 'FREE ENERGIE OF THE ION-ELECTRON SYSTEM' in line:
            # Last section before next ionic step
            nskip = 2
            parts = cl(lines[n + nskip]).strip().split()
            energy_free = float(parts[4])  # Force consistent

            nskip = 4
            parts = cl(lines[n + nskip]).strip().split()
            energy_zero = float(parts[6])  # Extrapolated to 0 K

            atoms.set_calculator(SinglePointCalculator(atoms,
                                                       energy=energy_zero,
                                                       free_energy=energy_free,
                                                       forces=forces,
                                                       stress=stress))
            if len(magnetization) > 0:
                mag = np.asarray(magnetization)
                atoms.calc.magmoms = mag
                atoms.calc.results['magmoms'] = mag
            if magmom is not None:
                atoms.calc.results['magmom'] = magmom
    return atoms


def _cl(line):
    """Auxiliary check line function.
    See issue #179, https://gitlab.com/ase/ase/issues/179
    Only call in cases we need the numeric values
    """
    if re.search('[0-9]-[0-9]', line):
        line = re.sub('([0-9])-([0-9])', r'\1 -\2', line)
    return line


def _read_outcar_header(fd):
    constr = None               # Should we re-implement this?

    cl = _cl                    # Aliasing

    species = []
    natoms = 0
    species_num = []
    symbols = []

    # Get atomic species
    for line in fd:
        line = line.strip()
        if 'POTCAR:' in line:
            temp = line.split()[2]
            for c in ['.', '_', '1']:
                if c in temp:
                    temp = temp[0:temp.find(c)]
            species += [temp]
        elif 'ions per type' in line:
            species = species[:len(species) // 2]
            parts = cl(line).split()
            ntypes = min(len(parts) - 4, len(species))
            for ispecies in range(ntypes):
                species_num += [int(parts[ispecies + 4])]
                natoms += species_num[-1]
                for iatom in range(species_num[-1]):
                    symbols += [species[ispecies]]
            return natoms, symbols, constr

    # Incomplete OUTCAR, we can't determine atoms
    raise IOError('Header data missing from OUTCAR')


def outcarchunks(fd):
    natoms, symbols, constr = _read_outcar_header(fd)  # First get header info

    while True:
        try:
            # Build chunck which contains 1 complete atoms object
            lines = []
            while True:
                line = next(fd)
                lines.append(line)
                if 'FREE ENERGIE OF THE ION-ELECTRON SYSTEM' in line:
                    # Add 4 more lines to include energy
                    for _ in range(4):
                        lines.append(next(fd))
                    break
        except StopIteration:
            # End of file
            return
        yield OUTCARChunck(lines, natoms, symbols, constr)


def iread_vasp_out(filename, index=-1):
    """Import OUTCAR type file, as a generator."""
    it = ImageIterator(outcarchunks)
    return it(filename, index=index)


@reader
def read_vasp_out(filename='OUTCAR', index=-1):
    """Import OUTCAR type file.

    Reads unitcell, atom positions, energies, and forces from the OUTCAR file
    and attempts to read constraints (if any) from CONTCAR/POSCAR, if present.
    """

    try:  # try to read constraints, first from CONTCAR, then from POSCAR
        constr = read_vasp('CONTCAR').constraints
    except Exception:
        # XXX: Which exceptions? OSError only?
        try:
            constr = read_vasp('POSCAR').constraints
        except Exception:
            # XXX: Which exceptions? OSError only?
            constr = None

    f = filename
    g = iread_vasp_out(f, index=index)
    if isinstance(index, (slice, basestring)):
        return list(g)
    else:
        return next(g)


@reader
def read_vasp_xdatcar(filename='XDATCAR', index=-1):
    """Import XDATCAR file

       Reads all positions from the XDATCAR and returns a list of
       Atoms objects.  Useful for viewing optimizations runs
       from VASP5.x

       Constraints ARE NOT stored in the XDATCAR, and as such, Atoms
       objects retrieved from the XDATCAR will not have constraints set.
    """

    import numpy as np
    from ase import Atoms

    f = filename
    images = list()

    cell = np.eye(3)
    atomic_formula = str()

    while True:
        comment_line = f.readline()
        if "Direct configuration=" not in comment_line:
            try:
                lattice_constant = float(f.readline())
            except Exception:
                # XXX: When would this happen?
                break

            xx = [float(x) for x in f.readline().split()]
            yy = [float(y) for y in f.readline().split()]
            zz = [float(z) for z in f.readline().split()]
            cell = np.array([xx, yy, zz]) * lattice_constant

            symbols = f.readline().split()
            numbers = [int(n) for n in f.readline().split()]
            total = sum(numbers)

            atomic_formula = str()
            for n, sym in enumerate(symbols):
                atomic_formula += '%s%s' % (sym, numbers[n])

            f.readline()

        coords = [np.array(f.readline().split(), np.float)
                  for ii in range(total)]

        image = Atoms(atomic_formula, cell=cell, pbc=True)
        image.set_scaled_positions(np.array(coords))
        images.append(image)

    if not index:
        return images
    else:
        return images[index]


def __get_xml_parameter(par):
    """An auxiliary function that enables convenient extraction of
    parameter values from a vasprun.xml file with proper type
    handling.

    """

    def to_bool(b):
        if b == 'T':
            return True
        else:
            return False

    to_type = {'int': int,
               'logical': to_bool,
               'string': str,
               'float': float}

    text = par.text
    if text is None:
        text = ''

    # Float parameters do not have a 'type' attrib
    var_type = to_type[par.attrib.get('type', 'float')]

    try:
        if par.tag == 'v':
            return list(map(var_type, text.split()))
        else:
            return var_type(text.strip())
    except ValueError:
        # Vasp can sometimes write "*****" due to overflow
        return None


def read_vasp_xml(filename='vasprun.xml', index=-1):
    """Parse vasprun.xml file.

    Reads unit cell, atom positions, energies, forces, and constraints
    from vasprun.xml file
    """

    import numpy as np
    import xml.etree.ElementTree as ET
    from ase import Atoms
    from ase.constraints import FixAtoms, FixScaled
    from ase.calculators.singlepoint import (SinglePointDFTCalculator,
                                             SinglePointKPoint)
    from ase.units import GPa
    from collections import OrderedDict

    tree = ET.iterparse(filename, events=['start', 'end'])

    atoms_init = None
    calculation = []
    ibz_kpts = None
    kpt_weights = None
    parameters = OrderedDict()

    try:
        for event, elem in tree:

            if event == 'end':
                if elem.tag == 'kpoints':
                    for subelem in elem.iter(tag='generation'):
                        kpts_params = OrderedDict()
                        parameters['kpoints_generation'] = kpts_params
                        for par in subelem.iter():
                            if par.tag in ['v', 'i']:
                                parname = par.attrib['name'].lower()
                                kpts_params[parname] = __get_xml_parameter(par)

                    kpts = elem.findall("varray[@name='kpointlist']/v")
                    ibz_kpts = np.zeros((len(kpts), 3))

                    for i, kpt in enumerate(kpts):
                        ibz_kpts[i] = [float(val) for val in kpt.text.split()]

                    kpt_weights = elem.findall('varray[@name="weights"]/v')
                    kpt_weights = [float(val.text) for val in kpt_weights]

                elif elem.tag == 'parameters':
                    for par in elem.iter():
                        if par.tag in ['v', 'i']:
                            parname = par.attrib['name'].lower()
                            parameters[parname] = __get_xml_parameter(par)

                elif elem.tag == 'atominfo':
                    species = []

                    for entry in elem.find("array[@name='atoms']/set"):
                        species.append(entry[0].text.strip())

                    natoms = len(species)

                elif (elem.tag == 'structure' and
                      elem.attrib.get('name') == 'initialpos'):
                    cell_init = np.zeros((3, 3), dtype=float)

                    for i, v in enumerate(elem.find(
                            "crystal/varray[@name='basis']")):
                        cell_init[i] = np.array([
                            float(val) for val in v.text.split()])

                    scpos_init = np.zeros((natoms, 3), dtype=float)

                    for i, v in enumerate(elem.find(
                            "varray[@name='positions']")):
                        scpos_init[i] = np.array([
                            float(val) for val in v.text.split()])

                    constraints = []
                    fixed_indices = []

                    for i, entry in enumerate(elem.findall(
                            "varray[@name='selective']/v")):
                        flags = (np.array(entry.text.split() ==
                                          np.array(['F', 'F', 'F'])))
                        if flags.all():
                            fixed_indices.append(i)
                        elif flags.any():
                            constraints.append(FixScaled(cell_init, i, flags))

                    if fixed_indices:
                        constraints.append(FixAtoms(fixed_indices))

                    atoms_init = Atoms(species,
                                       cell=cell_init,
                                       scaled_positions=scpos_init,
                                       constraint=constraints,
                                       pbc=True)

                elif elem.tag=='dipole':
                    dblock = elem.find('v[@name="dipole"]')
                    if dblock is not None:
                        dipole = np.array([float(val) for val in dblock.text.split()])

            elif event == 'start' and elem.tag == 'calculation':
                calculation.append(elem)

    except ET.ParseError as parse_error:
        if atoms_init is None:
            raise parse_error
        if calculation[-1].find('energy') is None:
            calculation = calculation[:-1]
        if not calculation:
            yield atoms_init

    if calculation:
        if isinstance(index, int):
            steps = [calculation[index]]
        else:
            steps = calculation[index]
    else:
        steps = []

    for step in steps:
        # Workaround for VASP bug, e_0_energy contains the wrong value
        # in calculation/energy, but calculation/scstep/energy does not
        # include classical VDW corrections. So, first calculate
        # e_0_energy - e_fr_energy from calculation/scstep/energy, then
        # apply that correction to e_fr_energy from calculation/energy.
        lastscf = step.findall('scstep/energy')[-1]
        try:
            lastdipole = step.findall('scstep/dipole')[-1]
        except:
            lastdipole = None

        de = (float(lastscf.find('i[@name="e_0_energy"]').text) -
              float(lastscf.find('i[@name="e_fr_energy"]').text))

        free_energy = float(step.find('energy/i[@name="e_fr_energy"]').text)
        energy = free_energy + de

        cell = np.zeros((3, 3), dtype=float)
        for i, vector in enumerate(step.find(
                'structure/crystal/varray[@name="basis"]')):
            cell[i] = np.array([float(val) for val in vector.text.split()])

        scpos = np.zeros((natoms, 3), dtype=float)
        for i, vector in enumerate(step.find(
                'structure/varray[@name="positions"]')):
            scpos[i] = np.array([float(val) for val in vector.text.split()])

        forces = None
        fblocks = step.find('varray[@name="forces"]')
        if fblocks is not None:
            forces = np.zeros((natoms, 3), dtype=float)
            for i, vector in enumerate(fblocks):
                forces[i] = np.array([float(val)
                                      for val in vector.text.split()])

        stress = None
        sblocks = step.find('varray[@name="stress"]')
        if sblocks is not None:
            stress = np.zeros((3, 3), dtype=float)
            for i, vector in enumerate(sblocks):
                stress[i] = np.array([float(val)
                                      for val in vector.text.split()])
            stress *= -0.1 * GPa
            stress = stress.reshape(9)[[0, 4, 8, 5, 2, 1]]

        dipole = None
        if lastdipole is not None:
            dblock = lastdipole.find('v[@name="dipole"]')
            if dblock is not None:
                dipole = np.zeros((1,3), dtype=float)
                dipole = np.array([float(val) for val in dblock.text.split()])

        dblock = step.find('dipole/v[@name="dipole"]')
        if dblock is not None:
            dipole = np.zeros((1,3), dtype=float)
            dipole = np.array([float(val) for val in dblock.text.split()])

        efermi = step.find('dos/i[@name="efermi"]')
        if efermi is not None:
            efermi = float(efermi.text)

        kpoints = []
        for ikpt in range(1, len(ibz_kpts) + 1):
            kblocks = step.findall(
                'eigenvalues/array/set/set/set[@comment="kpoint %d"]' % ikpt)
            if kblocks is not None:
                for spin, kpoint in enumerate(kblocks):
                    eigenvals = kpoint.findall('r')
                    eps_n = np.zeros(len(eigenvals))
                    f_n = np.zeros(len(eigenvals))
                    for j, val in enumerate(eigenvals):
                        val = val.text.split()
                        eps_n[j] = float(val[0])
                        f_n[j] = float(val[1])
                    if len(kblocks) == 1:
                        f_n *= 2
                    kpoints.append(SinglePointKPoint(kpt_weights[ikpt - 1],
                                                     spin, ikpt, eps_n, f_n))
        if len(kpoints) == 0:
            kpoints = None

        atoms = atoms_init.copy()
        atoms.set_cell(cell)
        atoms.set_scaled_positions(scpos)
        atoms.set_calculator(
            SinglePointDFTCalculator(atoms, energy=energy, forces=forces,
                                     stress=stress, free_energy=free_energy,
                                     ibzkpts=ibz_kpts,
                                     efermi=efermi, dipole=dipole))
        atoms.calc.name = 'vasp'
        atoms.calc.kpts = kpoints
        atoms.calc.parameters = parameters
        yield atoms


@writer
def write_vasp(filename, atoms, label='', direct=False, sort=None,
               symbol_count=None, long_format=True, vasp5=False,
               ignore_constraints=False):
    """Method to write VASP position (POSCAR/CONTCAR) files.

    Writes label, scalefactor, unitcell, # of various kinds of atoms,
    positions in cartesian or scaled coordinates (Direct), and constraints
    to file. Cartesian coordiantes is default and default label is the
    atomic species, e.g. 'C N H Cu'.
    """

    import numpy as np
    from ase.constraints import FixAtoms, FixScaled, FixedPlane, FixedLine

    f = filename

    if isinstance(atoms, (list, tuple)):
        if len(atoms) > 1:
            raise RuntimeError('Don\'t know how to save more than ' +
                               'one image to VASP input')
        else:
            atoms = atoms[0]

    # Check lattice vectors are finite
    if np.any(atoms.get_cell_lengths_and_angles() == 0.):
        raise RuntimeError(
            'Lattice vectors must be finite and not coincident. '
            'At least one lattice length or angle is zero.')

    # Write atom positions in scaled or cartesian coordinates
    if direct:
        coord = atoms.get_scaled_positions()
    else:
        coord = atoms.get_positions()

    constraints = atoms.constraints and not ignore_constraints

    if constraints:
        sflags = np.zeros((len(atoms), 3), dtype=bool)
        for constr in atoms.constraints:
            if isinstance(constr, FixScaled):
                sflags[constr.a] = constr.mask
            elif isinstance(constr, FixAtoms):
                sflags[constr.index] = [True, True, True]
            elif isinstance(constr, FixedPlane):
                mask = np.all(np.abs(np.cross(constr.dir, atoms.cell)) < 1e-5,
                              axis=1)
                if sum(mask) != 1:
                    raise RuntimeError(
                        'VASP requires that the direction of FixedPlane '
                        'constraints is parallel with one of the cell axis')
                sflags[constr.a] = mask
            elif isinstance(constr, FixedLine):
                mask = np.all(np.abs(np.cross(constr.dir, atoms.cell)) < 1e-5,
                              axis=1)
                if sum(mask) != 1:
                    raise RuntimeError(
                        'VASP requires that the direction of FixedLine '
                        'constraints is parallel with one of the cell axis')
                sflags[constr.a] = ~mask

    if sort:
        ind = np.argsort(atoms.get_chemical_symbols())
        symbols = np.array(atoms.get_chemical_symbols())[ind]
        coord = coord[ind]
        if constraints:
            sflags = sflags[ind]
    else:
        symbols = atoms.get_chemical_symbols()

    # Create a list sc of (symbol, count) pairs
    if symbol_count:
        sc = symbol_count
    else:
        sc = []
        psym = symbols[0]
        count = 0
        for sym in symbols:
            if sym != psym:
                sc.append((psym, count))
                psym = sym
                count = 1
            else:
                count += 1
        sc.append((psym, count))

    # Create the label
    if label == '':
        for sym, c in sc:
            label += '%2s ' % sym
    f.write(label + '\n')

    # Write unitcell in real coordinates and adapt to VASP convention
    # for unit cell
    # ase Atoms doesn't store the lattice constant separately, so always
    # write 1.0.
    f.write('%19.16f\n' % 1.0)
    if long_format:
        latt_form = ' %21.16f'
    else:
        latt_form = ' %11.6f'
    for vec in atoms.get_cell():
        f.write(' ')
        for el in vec:
            f.write(latt_form % el)
        f.write('\n')

    # If we're writing a VASP 5.x format POSCAR file, write out the
    # atomic symbols
    if vasp5:
        for sym, c in sc:
            f.write(' %3s' % sym)
        f.write('\n')

    # Numbers of each atom
    for sym, count in sc:
        f.write(' %3i' % count)
    f.write('\n')

    if constraints:
        f.write('Selective dynamics\n')

    if direct:
        f.write('Direct\n')
    else:
        f.write('Cartesian\n')

    if long_format:
        cform = ' %19.16f'
    else:
        cform = ' %9.6f'
    for iatom, atom in enumerate(coord):
        for dcoord in atom:
            f.write(cform % dcoord)
        if constraints:
            for flag in sflags[iatom]:
                if flag:
                    s = 'F'
                else:
                    s = 'T'
                f.write('%4s' % s)
        f.write('\n')
