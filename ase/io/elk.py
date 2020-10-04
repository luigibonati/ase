from pathlib import Path

import os
import numpy as np

from ase import Atoms
from ase.units import Bohr, Hartree
from ase.utils import reader, writer


elk_parameters = {'swidth': Hartree}


@reader
def read_elk(fd):
    """Import ELK atoms definition.

    Reads unitcell, atom positions, magmoms from elk.in/GEOMETRY.OUT file.
    """

    lines = fd.readlines()

    scale = np.ones(4)  # unit cell scale
    positions = []
    cell = []
    symbols = []
    magmoms = []
    periodic = np.array([True, True, True])
    # find cell scale
    for n, line in enumerate(lines):
        if line.split() == []:
            continue
        if line.strip() == 'scale':
            scale[0] = float(lines[n + 1])
        elif line.startswith('scale'):
            scale[int(line.strip()[-1])] = float(lines[n + 1])
    for n, line in enumerate(lines):
        if line.split() == []:
            continue
        if line.startswith('avec'):
            cell = np.array(
                [[float(v) * scale[1] for v in lines[n + 1].split()],
                 [float(v) * scale[2] for v in lines[n + 2].split()],
                 [float(v) * scale[3] for v in lines[n + 3].split()]])
        if line.startswith('atoms'):
            lines1 = lines[n + 1:]  # start subsearch
            spfname = []
            natoms = []
            atpos = []
            bfcmt = []
            for n1, line1 in enumerate(lines1):
                if line1.split() == []:
                    continue
                if 'spfname' in line1:
                    spfnamenow = lines1[n1].split()[0]
                    spfname.append(spfnamenow)
                    natomsnow = int(lines1[n1 + 1].split()[0])
                    natoms.append(natomsnow)
                    atposnow = []
                    bfcmtnow = []
                    for l in lines1[n1 + 2:n1 + 2 + natomsnow]:
                        atposnow.append([float(v) for v in l.split()[0:3]])
                        if len(l.split()) == 6:  # bfcmt present
                            bfcmtnow.append([float(v) for v in l.split()[3:]])
                    atpos.append(atposnow)
                    bfcmt.append(bfcmtnow)
    # symbols, positions, magmoms based on ELK spfname, atpos, and bfcmt
    symbols = ''
    positions = []
    magmoms = []
    for n, s in enumerate(spfname):
        symbols += str(s[1:].split('.')[0]) * natoms[n]
        positions += atpos[n]  # assumes fractional coordinates
        if len(bfcmt[n]) > 0:
            # how to handle cases of magmoms being one or three dim array?
            magmoms += [m[-1] for m in bfcmt[n]]
    atoms = Atoms(symbols, scaled_positions=positions, cell=[1, 1, 1])
    if len(magmoms) > 0:
        atoms.set_initial_magnetic_moments(magmoms)
    # final cell scale
    cell = cell * scale[0] * Bohr
    if periodic.any():
        atoms.set_cell(cell, scale_atoms=True)
        atoms.set_pbc(periodic)
    return atoms


@writer
def write_elk_in(fd, atoms, parameters=None):
    if parameters is None:
        parameters = {}

    parameters = dict(parameters)
    species_path = parameters.pop('species_dir')

    if 'xctype' in parameters:
        if 'xc' in parameters:
            raise RuntimeError("You can't use both 'xctype' and 'xc'!")

    if parameters.get('autokpt'):
        if 'kpts' in parameters:
            raise RuntimeError("You can't use both 'autokpt' and 'kpts'!")
        if 'ngridk' in parameters:
            raise RuntimeError(
                "You can't use both 'autokpt' and 'ngridk'!")
    if 'ngridk' in parameters:
        if 'kpts' in parameters:
            raise RuntimeError("You can't use both 'ngridk' and 'kpts'!")

    if parameters.get('autoswidth'):
        if 'smearing' in parameters:
            raise RuntimeError(
                "You can't use both 'autoswidth' and 'smearing'!")
        if 'swidth' in parameters:
            raise RuntimeError(
                "You can't use both 'autoswidth' and 'swidth'!")

    #fd = open(os.path.join(self.directory, 'elk.in'), 'w')

    # handle custom specifications of rmt
    # (absolute or relative to default) in Bohr
    # rmt = {'H': 0.7, 'O': -0.2, ...}

    if parameters.get('rmt', None) is not None:
        self.rmt = parameters['rmt'].copy()
        assert len(self.rmt.keys()) == len(list(set(self.rmt.keys()))), \
            'redundant rmt definitions'
        parameters.pop('rmt')  # this is not an elk keyword!
    else:
        pass
        # XXX self.rmt = None

    inp = {}
    inp.update(parameters)

    if 'xc' in parameters:
        xctype = {'LDA': 3,  # PW92
                  'PBE': 20,
                  'REVPBE': 21,
                  'PBESOL': 22,
                  'WC06': 26,
                  'AM05': 30,
                  'mBJLDA': (100, 208, 12)}[parameters['xc']]
        inp['xctype'] = xctype
        del inp['xc']

    if 'kpts' in parameters:
        mp = kpts2mp(atoms, parameters['kpts'])
        inp['ngridk'] = tuple(mp)
        vkloff = []  # is this below correct?
        for nk in mp:
            if nk % 2 == 0:  # shift kpoint away from gamma point
                vkloff.append(0.5)
            else:
                vkloff.append(0)
        inp['vkloff'] = vkloff
        del inp['kpts']

    if 'smearing' in parameters:
        name = parameters.smearing[0].lower()
        if name == 'methfessel-paxton':
            stype = parameters.smearing[2]
        else:
            stype = {'gaussian': 0,
                     'fermi-dirac': 3,
                     }[name]
        inp['stype'] = stype
        inp['swidth'] = parameters.smearing[1]
        del inp['smearing']

    # convert keys to ELK units
    for key, value in inp.items():
        if key in elk_parameters:
            inp[key] /= elk_parameters[key]

    # write all keys
    for key, value in inp.items():
        fd.write('%s\n' % key)
        if isinstance(value, bool):
            fd.write('.%s.\n\n' % ('false', 'true')[value])
        elif isinstance(value, (int, float)):
            fd.write('%s\n\n' % value)
        else:
            fd.write('%s\n\n' % ' '.join([str(x) for x in value]))

    # cell
    fd.write('avec\n')
    for vec in atoms.cell:
        fd.write('%.14f %.14f %.14f\n' % tuple(vec / Bohr))
    fd.write('\n')

    # atoms
    species = {}
    symbols = []
    for a, (symbol, m) in enumerate(
        zip(atoms.get_chemical_symbols(),
            atoms.get_initial_magnetic_moments())):
        if symbol in species:
            species[symbol].append((a, m))
        else:
            species[symbol] = [(a, m)]
            symbols.append(symbol)
    fd.write('atoms\n%d\n' % len(species))
    # scaled = atoms.get_scaled_positions(wrap=False)
    scaled = np.linalg.solve(atoms.cell.T, atoms.positions.T).T
    for symbol in symbols:
        fd.write("'%s.in' : spfname\n" % symbol)
        fd.write('%d\n' % len(species[symbol]))
        for a, m in species[symbol]:
            fd.write('%.14f %.14f %.14f 0.0 0.0 %.14f\n' %
                     (tuple(scaled[a]) + (m,)))

    # custom species definitions

    if 0:
    #if self.rmt is not None:
        fd.write("\n")
        sfile = os.path.join(os.environ['ELK_SPECIES_PATH'], 'elk.in')
        assert os.path.exists(sfile)
        slines = open(sfile, 'r').readlines()
        # remove unused species
        for s in self.rmt.keys():
            if s not in species.keys():
                self.rmt.pop(s)
        # add undefined species with defaults
        for s in species.keys():
            if s not in self.rmt.keys():
                # use default rmt for undefined species
                self.rmt.update({s: 0.0})
        # write custom species into elk.in
        skeys = list(set(self.rmt.keys()))  # unique
        skeys.sort()
        for s in skeys:
            found = False
            for n, line in enumerate(slines):
                if line.find("'" + s + "'") > -1:
                    begline = n - 1
            for n, line in enumerate(slines[begline:]):
                if not line.strip():  # first empty line
                    endline = n
                    found = True
                    break
            assert found
            fd.write("species\n")
            # set rmt on third line
            rmt = self.rmt[s]
            assert isinstance(rmt, (float, int))
            if rmt <= 0.0:  # relative
                # split needed because H is defined with comments
                newrmt = (float(slines[begline + 3].split()[0].strip()) +
                          rmt)
            else:
                newrmt = rmt
            slines[begline + 3] = '%6s\n' % str(newrmt)
            for l in slines[begline: begline + endline]:
                fd.write('%s' % l)
            fd.write('\n')
    else:
        # use default species
        # if sppath is present in elk.in it overwrites species blocks!
        #species_path = os.environ['ELK_SPECIES_PATH']

        # Elk seems to concatenate path and filename in such a way
        # that we must put a / at the end:
        fd.write(f"sppath\n'{species_path}/'\n\n")
