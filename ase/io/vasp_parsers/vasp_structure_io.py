"""
=======
Module for reading and writing structure files for VASP.
POSCAR, CONTCAR
=======
"""
import numpy as np
import re
from ase import Atoms
from ase.utils import reader, writer
from ase.io import ParseError
from pathlib import Path

__all__ = [
    'read_vasp_structure', 'write_vasp_structure'
]

@reader
def read_vasp_structure(filename='CONTCAR'):
    """Import POSCAR/CONTCAR type file.

    Reads unitcell, atom positions and constraints from the POSCAR/CONTCAR
    file and tries to read atom types from POSCAR/CONTCAR header, if this fails
    the atom types are read from OUTCAR or POTCAR file.
    """

    from ase.constraints import FixAtoms, FixScaled
    from ase.data import chemical_symbols

    fd = filename
    
    fillist = fd.readlines()
    
    fd.seek(0)
    
    # The first line is in principle a comment line, however in VASP
    # 4.x a common convention is to have it contain the atom symbols,
    # eg. "Ag Ge" in the same order as later in the file (and POTCAR
    # for the full vasp run). In the VASP 5.x format this information
    # is found on the fifth line. Thus we save the first line and use
    # it in case we later detect that we're reading a VASP 4.x format
    # file.
    
    line1 = fd.readline()
    
    lattice_constant = float(fd.readline().split()[0])

    # Now the lattice vectors
    a = []
    for ii in range(3):
        s = fd.readline().split()
        floatvect = float(s[0]), float(s[1]), float(s[2])
        a.append(floatvect)

    basis_vectors = np.array(a) * lattice_constant

    # Number of atoms. Again this must be in the same order as
    # in the first line
    # or in the POTCAR or OUTCAR file
    atom_symbols = []
    numofatoms = fd.readline().split()
    # Check whether we have a VASP 4.x or 5.x format file. If the
    # format is 5.x, use the fifth line to provide information about
    # the atomic symbols.
    vasp5 = False
    try:
        int(numofatoms[0])
    except ValueError:
        vasp5 = True
        atomtypes = numofatoms
        numofatoms = fd.readline().split()

    # check for comments in numofatoms line and get rid of them if necessary
    commentcheck = np.array(['!' in s for s in numofatoms])
    if commentcheck.any():
        # only keep the elements up to the first including a '!':
        numofatoms = numofatoms[:np.arange(len(numofatoms))[commentcheck][0]]

    if not vasp5:
        # Split the comment line (first in the file) into words and
        # try to compose a list of chemical symbols
        from ase.formula import Formula
        atomtypes = []
        for word in line1.split():
            word_without_delims = re.sub(r"-|_|,|\.|=|[0-9]|^", "", word)
            if len(word_without_delims) < 1:
                continue
            try:
                atomtypes.extend(list(Formula(word_without_delims)))
            except ValueError:
                # print(atomtype, e, 'is comment')
                pass
        # Now the list of chemical symbols atomtypes must be formed.
        # For example: atomtypes = ['Pd', 'C', 'O']
        
        numsyms = len(numofatoms)
        if len(atomtypes) < numsyms:
            # First line in POSCAR/CONTCAR didn't contain enough symbols.
            
            #Check for AFLOW style POSCAR file.
            
            if len(fillist[7].split())==4:   
                try:
                    atomtypes=[]
                    for i in range(7, len(fillist)):
                        atomtypes.extend(list(Formula(fillist[i].split()[3])))
                    atomtypes = list(dict.fromkeys(atomtypes))
                except ValueError:
                    # print(atomtype, e, 'is comment')
                    pass
                
            
            # Sometimes the first line in POSCAR/CONTCAR is of the form
            # "CoP3_In-3.pos". Check for this case and extract atom types
            elif len(atomtypes) == 1 and '_' in atomtypes[0]:
                atomtypes = get_atomtypes_from_formula(atomtypes[0])
            else:
                atomtypes = atomtypes_outpot(fd.name, numsyms)
        else:
            try:
                for atype in atomtypes[:numsyms]:
                    if atype not in chemical_symbols:
                        raise KeyError
            except KeyError:
                atomtypes = atomtypes_outpot(fd.name, numsyms)

    for i, num in enumerate(numofatoms):
        numofatoms[i] = int(num)
        [atom_symbols.append(atomtypes[i]) for na in range(numofatoms[i])]

    # Check if Selective dynamics is switched on
    sdyn = fd.readline()
    selective_dynamics = sdyn[0].lower() == 's'

    # Check if atom coordinates are cartesian or direct
    if selective_dynamics:
        ac_type = fd.readline()
    else:
        ac_type = sdyn
    cartesian = ac_type[0].lower() == 'c' or ac_type[0].lower() == 'k'
    tot_natoms = sum(numofatoms)
    atoms_pos = np.empty((tot_natoms, 3))
    if selective_dynamics:
        selective_flags = np.empty((tot_natoms, 3), dtype=bool)
    for atom in range(tot_natoms):
        ac = fd.readline().split()
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


def get_atomtypes(fname):
    """Given a file name, get the atomic symbols.

    The function can get this information from OUTCAR and POTCAR
    format files.  The files can also be compressed with gzip or
    bzip2.

    """
    fpath = Path(fname)

    atomtypes = []
    atomtypes_alt = []
    if fpath.suffix == '.gz':
        import gzip
        opener = gzip.open
    elif fpath.suffix == '.bz2':
        import bz2
        opener = bz2.BZ2File
    else:
        opener = open
    with opener(fpath) as fd:
        for line in fd:
            if 'TITEL' in line:
                atomtypes.append(line.split()[3].split('_')[0].split('.')[0])
            elif 'POTCAR:' in line:
                atomtypes_alt.append(line.split()[2].split('_')[0].split('.')[0])

    if len(atomtypes) == 0 and len(atomtypes_alt) > 0:
        # old VASP doesn't echo TITEL, but all versions print out species lines
        # preceded by "POTCAR:", twice
        if len(atomtypes_alt) % 2 != 0:
            raise ParseError(f'Tried to get atom types from {len(atomtypes_alt)} "POTCAR": '
                              'lines in OUTCAR, but expected an even number')
        atomtypes = atomtypes_alt[0:len(atomtypes_alt)//2]

    return atomtypes


def atomtypes_outpot(posfname, numsyms):
    """Try to retrieve chemical symbols from OUTCAR or POTCAR

    If getting atomtypes from the first line in POSCAR/CONTCAR fails, it might
    be possible to find the data in OUTCAR or POTCAR, if these files exist.

    posfname -- The filename of the POSCAR/CONTCAR file we're trying to read

    numsyms -- The number of symbols we must find

    """
    posfpath = Path(posfname)

    # Check files with exactly same path except POTCAR/OUTCAR instead
    # of POSCAR/CONTCAR.
    fnames = [posfpath.with_name('POTCAR'),
              posfpath.with_name('OUTCAR')]
    # Try the same but with compressed files
    fsc = []
    for fnpath in fnames:
        fsc.append(fnpath.parent / (fnpath.name + '.gz'))
        fsc.append(fnpath.parent / (fnpath.name + '.bz2'))
    for f in fsc:
        fnames.append(f)
    # Code used to try anything with POTCAR or OUTCAR in the name
    # but this is no longer supported

    tried = []
    for fn in fnames:
        if fn in posfpath.parent.iterdir():
            tried.append(fn)
            at = get_atomtypes(fn)
            if len(at) == numsyms:
                return at

    raise ParseError('Could not determine chemical symbols. Tried files ' +
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

@writer
def write_vasp_structure(filename,
                         atoms,
                         direct=False,
                         sort=True,
                         symbol_count=None,
                         vasp5=True,
                         ignore_constraints=False,
                         wrap=False):
    """Method to write VASP position (POSCAR/CONTCAR) files.

    Writes label, scalefactor, unitcell, # of various kinds of atoms,
    positions in cartesian or scaled coordinates (Direct), and constraints
    to file. Cartesian coordinates is default and default label is the
    atomic species, e.g. 'C N H Cu'.
    """

    from ase.constraints import FixAtoms, FixScaled, FixedPlane, FixedLine

    fd = filename  # @writer decorator ensures this arg is a file descriptor

    if isinstance(atoms, (list, tuple)):
        if len(atoms) > 1:
            raise RuntimeError('Don\'t know how to save more than ' +
                               'one image to VASP input')
        else:
            atoms = atoms[0]

    # Check lattice vectors are finite
    if np.any(atoms.cell.cellpar() == 0.):
        raise RuntimeError(
            'Lattice vectors must be finite and not coincident. '
            'At least one lattice length or angle is zero.')

    # Write atom positions in scaled or cartesian coordinates
    if direct:
        coord = atoms.get_scaled_positions(wrap=wrap)
    else:
        coord = atoms.get_positions(wrap=wrap)

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
        sc = _symbol_count_from_symbols(symbols)

    # Create the label
    label = ''
    for sym, c in sc:
        label += f'{sym:2s} '
    fd.write(label + '\n')

    # Write unitcell in real coordinates and adapt to VASP convention
    # for unit cell
    # ase Atoms doesn't store the lattice constant separately, so always
    # write 1.0.
    fd.write(f'{1.0:19.16f}\n')
    
    for vec in atoms.get_cell():
        fd.write(' ')
        for el in vec: 
            fd.write(f' {el:21.16f}')
        fd.write('\n')

    # Write out symbols (if VASP 5.x) and counts of atoms
    _write_symbol_count(fd, sc, vasp5=vasp5)

    if constraints:
        fd.write('Selective dynamics\n')

    if direct:
        fd.write('Direct\n')
    else:
        fd.write('Cartesian\n')

    for iatom, atom in enumerate(coord):
        for dcoord in atom:
            fd.write(f' {dcoord:19.16f}')
        if constraints:
            for flag in sflags[iatom]:
                if flag:
                    s = 'F'
                else:
                    s = 'T'
                fd.write(f'{s:4s}')
        fd.write('\n')


def _symbol_count_from_symbols(symbols):
    """Reduce list of chemical symbols into compact VASP notation

    args:
        symbols (iterable of str)

    returns:
        list of pairs [(el1, c1), (el2, c2), ...]
    """
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
    return sc


def _write_symbol_count(fd, sc, vasp5=True):
    """Write the symbols and numbers block for POSCAR or XDATCAR

    Args:
        f (fd): Descriptor for writable file
        sc (list of 2-tuple): list of paired elements and counts
        vasp5 (bool): if False, omit symbols and only write counts

    e.g. if sc is [(Sn, 4), (S, 6)] then write::

      Sn   S
       4   6

    """
    if vasp5:
        for sym, _ in sc:
            fd.write(f' {sym:3s}')
        fd.write('\n')

    for _, count in sc:
        fd.write(f' {count:3d}')
    fd.write('\n')
