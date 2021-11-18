import os
import numpy as np
import ase.io
from ase.atoms import Atoms
from ase.calculators.calculator import kpts2ndarray, kpts2sizeandoffsets
from ase.utils import reader, writer
from ase.units import Hartree, Bohr


def prepare_dftb_input(outfile, atoms, parameters, properties, directory):
    """ Write the innput file for the dftb+ calculation.
        Geometry is taken always from the file 'geo_end.gen'.
    """
    # Write geometry information
    ase.io.write(f'{directory}/geo_end.gen', atoms)

    outfile.write('Geometry = GenFormat { \n')
    outfile.write('    <<< "geo_end.gen" \n')
    outfile.write('} \n')
    outfile.write(' \n')

    # TODO: this is just a hack. Need to be cleaned up!
    params = parameters.copy()

    assert 'skt_path' in params.keys()
    slako_dir = params.pop('skt_path')

    pcpot = params.pop('pcpot') if 'pcpot' in params else None  # TODO: What does this do, needed?

    do_forces = False
    if properties is not None:
        if 'forces' in properties or 'stress' in properties:
            do_forces = True

    kpts = params.pop('kpts') if 'kpts' in params else None
    params.update(dict(
            Hamiltonian_='DFTB',
            Hamiltonian_SlaterKosterFiles_='Type2FileNames',
            Hamiltonian_SlaterKosterFiles_Prefix=slako_dir,
            Hamiltonian_SlaterKosterFiles_Separator='"-"',
            Hamiltonian_SlaterKosterFiles_Suffix='".skf"',
            Hamiltonian_MaxAngularMomentum_='',
            Options_='',
            Options_WriteResultsTag='Yes')
        )

    if kpts is not None:
        initkey = 'Hamiltonian_KPointsAndWeights'
        mp_mesh = None
        offsets = None

        if isinstance(kpts, dict):
            if 'path' in kpts:
                # kpts is path in Brillouin zone
                params[initkey + '_'] = 'Klines '
                kpts_coord = kpts2ndarray(kpts, atoms=atoms)
            else:
                # kpts is (implicit) definition of
                # Monkhorst-Pack grid
                params[initkey + '_'] = 'SupercellFolding '
                mp_mesh, offsets = kpts2sizeandoffsets(atoms=atoms,
                                                       **kpts)
        elif np.array(kpts).ndim == 1:
            # kpts is Monkhorst-Pack grid
            params[initkey + '_'] = 'SupercellFolding '
            mp_mesh = kpts
            offsets = [0.] * 3
        elif np.array(kpts).ndim == 2:
            # kpts is (N x 3) list/array of k-point coordinates
            # each will be given equal weight
            params[initkey + '_'] = ''
            kpts_coord = np.array(kpts)
        else:
            raise ValueError('Illegal kpts definition:' + str(kpts))

        if mp_mesh is not None:
            eps = 1e-10
            for i in range(3):
                key = initkey + '_empty%03d' % i
                val = [mp_mesh[i] if j == i else 0 for j in range(3)]
                params[key] = ' '.join(map(str, val))
                offsets[i] *= mp_mesh[i]
                assert abs(offsets[i]) < eps or abs(offsets[i] - 0.5) < eps
                # DFTB+ uses a different offset convention, where
                # the k-point mesh is already Gamma-centered prior
                # to the addition of any offsets
                if mp_mesh[i] % 2 == 0:
                    offsets[i] += 0.5
            key = initkey + '_empty%03d' % 3
            params[key] = ' '.join(map(str, offsets))

        elif kpts_coord is not None:
            for i, c in enumerate(kpts_coord):
                key = initkey + '_empty%09d' % i
                c_str = ' '.join(map(str, c))
                if 'Klines' in params[initkey + '_']:
                    c_str = '1 ' + c_str
                else:
                    c_str += ' 1.0'
                params[key] = c_str

    s = 'Hamiltonian_MaxAngularMomentum_'
    for key in params:
        if key.startswith(s) and len(key) > len(s):
            break
    else:
        # User didn't specify max angular mometa.  Get them from
        # the .skf files:
        symbols = set(atoms.get_chemical_symbols())
        for symbol in symbols:
            path = os.path.join(slako_dir,
                                '{0}-{0}.skf'.format(symbol))
            l = read_max_angular_momentum(path)
            params[s + symbol] = '"{}"'.format('spdf'[l])

    # --------MAIN KEYWORDS-------
    previous_key = 'dummy_'
    myspace = ' '
    for key, value in sorted(params.items()):
        current_depth = key.rstrip('_').count('_')
        previous_depth = previous_key.rstrip('_').count('_')
        for my_backsclash in reversed(
                range(previous_depth - current_depth)):
            outfile.write(3 * (1 + my_backsclash) * myspace + '} \n')
        outfile.write(3 * current_depth * myspace)
        if key.endswith('_') and len(value) > 0:
            outfile.write(key.rstrip('_').rsplit('_')[-1] +
                          ' = ' + str(value) + '{ \n')
        elif (key.endswith('_') and (len(value) == 0)
              and current_depth == 0):  # E.g. 'Options {'
            outfile.write(key.rstrip('_').rsplit('_')[-1] +
                          ' ' + str(value) + '{ \n')
        elif (key.endswith('_') and (len(value) == 0)
              and current_depth > 0):  # E.g. 'Hamiltonian_Max... = {'
            outfile.write(key.rstrip('_').rsplit('_')[-1] +
                          ' = ' + str(value) + '{ \n')
        elif key.count('_empty') == 1:
            outfile.write(str(value) + ' \n')
        elif ((key == 'Hamiltonian_ReadInitialCharges') and
              (str(value).upper() == 'YES')):
            f1 = os.path.isfile(directory + os.sep + 'charges.dat')
            f2 = os.path.isfile(directory + os.sep + 'charges.bin')
            if not (f1 or f2):
                print('charges.dat or .bin not found, switching off guess')
                value = 'No'
            outfile.write(key.rsplit('_')[-1] + ' = ' + str(value) + ' \n')
        else:
            outfile.write(key.rsplit('_')[-1] + ' = ' + str(value) + ' \n')
        if pcpot is not None and ('DFTB' in str(value)):
            outfile.write('   ElectricField = { \n')
            outfile.write('      PointCharges = { \n')
            outfile.write(
                '         CoordsAndCharges [Angstrom] = DirectRead { \n')
            outfile.write('            Records = ' +
                          str(len(pcpot.mmcharges)) + ' \n')
            outfile.write(
                '            File = "dftb_external_charges.dat" \n')
            outfile.write('         } \n')
            outfile.write('      } \n')
            outfile.write('   } \n')
        previous_key = key
    current_depth = key.rstrip('_').count('_')
    for my_backsclash in reversed(range(current_depth)):
        outfile.write(3 * my_backsclash * myspace + '} \n')
    outfile.write('ParserOptions { \n')
    outfile.write('   IgnoreUnprocessedNodes = Yes  \n')
    outfile.write('} \n')
    if do_forces:
        outfile.write('Analysis { \n')
        outfile.write('   CalculateForces = Yes  \n')
        outfile.write('} \n')

def read_dftb_outputs(directory, label):
    """ all results are read from results.tag file
        It will be destroyed after it is read to avoid
        reading it once again after some runtime error """
    results = {}

    with open(os.path.join(directory, 'results.tag'), 'r') as fd:
        lines = fd.readlines()

    do_forces = False
    for li in lines:
        if li.startswith('forces'):
            do_forces = True

    with open(f'{directory}/{label}_pin.hsd', 'r') as fd:
        results['atoms'] = read_dftb(fd)

    charges, energy, dipole = read_charges_energy_dipole(directory, len(results['atoms']))
    if charges is not None:
        results['charges'] = charges
    results['energy'] = energy
    if dipole is not None:
        results['dipole'] = dipole
    if do_forces:
        forces = read_forces(lines)
        results['forces'] = forces
    mmpositions = None

    stress = read_stress(lines)
    if stress is not None:
        results['stress'] = stress


    # eigenvalues and fermi levels
    fermi_levels = read_fermi_levels(lines)
    if fermi_levels is not None:
        results['fermi_levels'] = fermi_levels

    eigenvalues = read_eigenvalues(lines)
    if eigenvalues is not None:
        results['eigenvalues'] = eigenvalues

    # calculation was carried out with atoms written in write_input
    os.remove(os.path.join(directory, 'results.tag'))

    return results

def read_max_angular_momentum(path):
    """Read maximum angular momentum from .skf file.

    See dftb.org for A detailed description of the Slater-Koster file format.
    """
    with open(path, 'r') as fd:
        line = fd.readline()
        if line[0] == '@':
            # Extended format
            fd.readline()
            l = 3
            pos = 9
        else:
            # Simple format:
            l = 2
            pos = 7

        # Sometimes there ar commas, sometimes not:
        line = fd.readline().replace(',', ' ')

        occs = [float(f) for f in line.split()[pos:pos + l + 1]]
        for f in occs:
            if f > 0.0:
                return l
            l -= 1

def read_charges_energy_dipole(directory, num_atoms):
    """Get partial charges on atoms
        in case we cannot find charges they are set to None
    """
    with open(os.path.join(directory, 'detailed.out'), 'r') as fd:
        lines = fd.readlines()

    for line in lines:
        if line.strip().startswith('Total energy:'):
            energy = float(line.split()[2]) * Hartree
            break

    qm_charges = []
    for n, line in enumerate(lines):
        if ('Atom' and 'Charge' in line):
            chargestart = n + 1
            break
    else:
        # print('Warning: did not find DFTB-charges')
        # print('This is ok if flag SCC=No')
        return None, energy, None

    lines1 = lines[chargestart:(chargestart + num_atoms)]
    for line in lines1:
        qm_charges.append(float(line.split()[-1]))

    dipole = None
    for line in lines:
        if 'Dipole moment:' in line and 'au' in line:
            words = line.split()
            dipole = np.array(
                [float(w) for w in words[-4:-1]]) * Bohr

    return np.array(qm_charges), energy, dipole

def read_forces(lines):  # TODO: pass lines (?), as self.lines is not available
    """Read Forces from dftb output file (results.tag)."""
    from ase.units import Hartree, Bohr

    # Initialise the indices so their scope
    # reaches outside of the for loop
    index_force_begin = -1
    index_force_end = -1

    # Force line indexes
    for iline, line in enumerate(lines):
        fstring = 'forces   '
        if line.find(fstring) >= 0:
            index_force_begin = iline + 1
            line1 = line.replace(':', ',')
            index_force_end = iline + 1 + \
                int(line1.split(',')[-1])
            break

    gradients = []
    for j in range(index_force_begin, index_force_end):
        word = lines[j].split()
        gradients.append([float(word[k]) for k in range(0, 3)])

    return np.array(gradients) * Hartree / Bohr

def read_fermi_levels(lines):
    """ Read Fermi level(s) from dftb output file (results.tag). """
    # Fermi level line indexes
    for iline, line in enumerate(lines):
        fstring = 'fermi_level   '
        if line.find(fstring) >= 0:
            index_fermi = iline + 1
            break
    else:
        return None

    fermi_levels = []
    words = lines[index_fermi].split()
    assert len(words) in [1, 2], 'Expected either 1 or 2 Fermi levels'

    for word in words:
        e = float(word)
        # In non-spin-polarized calculations with DFTB+ v17.1,
        # two Fermi levels are given, with the second one being 0,
        # but we don't want to add that one to the list
        if abs(e) > 1e-8:
            fermi_levels.append(e)

    return np.array(fermi_levels) * Hartree

def read_eigenvalues(lines):
    """ Read Eigenvalues from dftb output file (results.tag).
        Unfortunately, the order seems to be scrambled. """
    # Eigenvalue line indexes
    index_eig_begin = None
    for iline, line in enumerate(lines):
        fstring = 'eigenvalues   '
        if line.find(fstring) >= 0:
            index_eig_begin = iline + 1
            line1 = line.replace(':', ',')
            ncol, nband, nkpt, nspin = map(int, line1.split(',')[-4:])
            break
    else:
        return None

    # Take into account that the last row may lack
    # columns if nkpt * nspin * nband % ncol != 0
    nrow = int(np.ceil(nkpt * nspin * nband * 1. / ncol))
    index_eig_end = index_eig_begin + nrow
    ncol_last = len(lines[index_eig_end - 1].split())
    lines[index_eig_end - 1] += ' 0.0 ' * (ncol - ncol_last)

    eig = np.loadtxt(lines[index_eig_begin:index_eig_end]).flatten()
    eig *= Hartree
    N = nkpt * nband
    eigenvalues = [eig[i * N:(i + 1) * N].reshape((nkpt, nband))
                   for i in range(nspin)]

    return eigenvalues

def read_stress(lines):
    """Read stress from dftb output file (results.tag)."""
    sstring = 'stress'
    have_stress = False
    stress = list()
    for iline, line in enumerate(lines):
        if sstring in line:
            have_stress = True
            start = iline + 1
            end = start + 3
            for i in range(start, end):
                cell = [float(x) for x in lines[i].split()]
                stress.append(cell)
    if have_stress:
        stress = -np.array(stress) * Hartree / Bohr**3
        return stress.flat[[0, 4, 8, 5, 2, 1]]
    else:
        return None




















@reader
def read_dftb(fd):
    """Method to read coordinates from the Geometry section
    of a DFTB+ input file (typically called "dftb_in.hsd").

    As described in the DFTB+ manual, this section can be
    in a number of different formats. This reader supports
    the GEN format and the so-called "explicit" format.

    The "explicit" format is unique to DFTB+ input files.
    The GEN format can also be used in a stand-alone fashion,
    as coordinate files with a `.gen` extension. Reading and
    writing such files is implemented in `ase.io.gen`.
    """
    lines = fd.readlines()

    atoms_pos = []
    atom_symbols = []
    type_names = []
    my_pbc = False
    fractional = False
    mycell = []

    for iline, line in enumerate(lines):
        if line.strip().startswith('#'):
            pass
        elif 'genformat' in line.lower():
            natoms = int(lines[iline + 1].split()[0])
            if lines[iline + 1].split()[1].lower() == 's':
                my_pbc = True
            elif lines[iline + 1].split()[1].lower() == 'f':
                my_pbc = True
                fractional = True

            symbols = lines[iline + 2].split()

            for i in range(natoms):
                index = iline + 3 + i
                aindex = int(lines[index].split()[1]) - 1
                atom_symbols.append(symbols[aindex])

                position = [float(p) for p in lines[index].split()[2:]]
                atoms_pos.append(position)

            if my_pbc:
                for i in range(3):
                    index = iline + 4 + natoms + i
                    cell = [float(c) for c in lines[index].split()]
                    mycell.append(cell)
        else:
            if 'TypeNames' in line:
                col = line.split()
                for i in range(3, len(col) - 1):
                    type_names.append(col[i].strip("\""))
            elif 'Periodic' in line:
                if 'Yes' in line:
                    my_pbc = True
            elif 'LatticeVectors' in line:
                for imycell in range(3):
                    extraline = lines[iline + imycell + 1]
                    cols = extraline.split()
                    mycell.append(
                        [float(cols[0]), float(cols[1]), float(cols[2])])
            else:
                pass

    if not my_pbc:
        mycell = [0.] * 3

    start_reading_coords = False
    stop_reading_coords = False
    for line in lines:
        if line.strip().startswith('#'):
            pass
        else:
            if 'TypesAndCoordinates' in line:
                start_reading_coords = True
            if start_reading_coords:
                if '}' in line:
                    stop_reading_coords = True
            if (start_reading_coords and not stop_reading_coords
                and 'TypesAndCoordinates' not in line):
                typeindexstr, xxx, yyy, zzz = line.split()[:4]
                typeindex = int(typeindexstr)
                symbol = type_names[typeindex - 1]
                atom_symbols.append(symbol)
                atoms_pos.append([float(xxx), float(yyy), float(zzz)])

    if fractional:
        atoms = Atoms(scaled_positions=atoms_pos, symbols=atom_symbols,
                      cell=mycell, pbc=my_pbc)
    elif not fractional:
        atoms = Atoms(positions=atoms_pos, symbols=atom_symbols,
                      cell=mycell, pbc=my_pbc)

    return atoms


def read_dftb_velocities(atoms, filename):
    """Method to read velocities (AA/ps) from DFTB+ output file geo_end.xyz
    """
    from ase.units import second
    # AA/ps -> ase units
    AngdivPs2ASE = 1.0 / (1e-12 * second)

    with open(filename) as fd:
        lines = fd.readlines()

    # remove empty lines
    lines_ok = []
    for line in lines:
        if line.rstrip():
            lines_ok.append(line)

    velocities = []
    natoms = len(atoms)
    last_lines = lines_ok[-natoms:]
    for iline, line in enumerate(last_lines):
        inp = line.split()
        velocities.append([float(inp[5]) * AngdivPs2ASE,
                           float(inp[6]) * AngdivPs2ASE,
                           float(inp[7]) * AngdivPs2ASE])

    atoms.set_velocities(velocities)
    return atoms


@reader
def read_dftb_lattice(fileobj, images=None):
    """Read lattice vectors from MD and return them as a list.

    If a molecules are parsed add them there."""
    if images is not None:
        append = True
        if hasattr(images, 'get_positions'):
            images = [images]
    else:
        append = False

    fileobj.seek(0)
    lattices = []
    for line in fileobj:
        if 'Lattice vectors' in line:
            vec = []
            for i in range(3):  # DFTB+ only supports 3D PBC
                line = fileobj.readline().split()
                try:
                    line = [float(x) for x in line]
                except ValueError:
                    raise ValueError('Lattice vector elements should be of '
                                     'type float.')
                vec.extend(line)
            lattices.append(np.array(vec).reshape((3, 3)))

    if append:
        if len(images) != len(lattices):
            raise ValueError('Length of images given does not match number of '
                             'cell vectors found')

        for i, atoms in enumerate(images):
            atoms.set_cell(lattices[i])
            # DFTB+ only supports 3D PBC
            atoms.set_pbc(True)
        return
    else:
        return lattices


@writer
def write_dftb(fileobj, images):
    """Write structure in GEN format (refer to DFTB+ manual).
       Multiple snapshots are not allowed. """
    from ase.io.gen import write_gen
    write_gen(fileobj, images)


def write_dftb_velocities(atoms, filename):
    """Method to write velocities (in atomic units) from ASE
       to a file to be read by dftb+
    """
    from ase.units import AUT, Bohr
    # ase units -> atomic units
    ASE2au = Bohr / AUT

    with open(filename, 'w') as fd:
        velocities = atoms.get_velocities()
        for velocity in velocities:
            fd.write(' %19.16f %19.16f %19.16f \n'
                     % (velocity[0] / ASE2au,
                        velocity[1] / ASE2au,
                        velocity[2] / ASE2au))
