from pathlib import Path
import re

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

    atoms.set_cell(cell, scale_atoms=True)
    atoms.pbc = True
    return atoms


@writer
def write_elk_in(fd, atoms, parameters=None):
    if parameters is None:
        parameters = {}

    parameters = dict(parameters)
    species_path = parameters.pop('species_dir', None)

    if parameters.get('spinpol') is None:
        if atoms.get_initial_magnetic_moments().any():
            parameters['spinpol'] = True

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

    # if sppath is present in elk.in it overwrites species blocks!
    #species_path = os.environ['ELK_SPECIES_PATH']

    # Elk seems to concatenate path and filename in such a way
    # that we must put a / at the end:
    if species_path is not None:
        fd.write(f"sppath\n'{species_path}/'\n\n")


class ElkReader:
    def __init__(self, path):
        self.path = Path(path)

    def read_everything(self):
        #converged = self.read_convergence()
        #if not converged:
        #    raise RuntimeError(f'ELK did not converge! Check {self.out}')
        yield from self.read_energy()
        #if self.parameters.get('tforce'):
        yield 'forces', self.read_forces()

        #yield 'width', self.read_electronic_temperature()
        #yield 'nbands', self.read_number_of_bands()
        #yield 'nelect', self.read_number_of_electrons()
        #yield 'niter', self.read_number_of_iterations()
        #yield 'magnetic_moment', self.read_magnetic_moment()
        yield from self.read_eigval()

    @property
    def out(self):
        return self.path / 'INFO.OUT'

    def read_energy(self):
        txt = (self.path / 'TOTENERGY.OUT').read_text()
        tokens = txt.split()
        energy = float(tokens[-1]) * Hartree
        yield 'free_energy', energy
        yield 'energy', energy

    def read_forces(self):
        lines = self.out.read_text().splitlines()
        forces = []
        for line in lines:
            if line.rfind('total force') > -1:
                forces.append(np.array([float(f)
                                        for f in line.split(':')[1].split()]))
        return np.array(forces) * Hartree / Bohr

    #def read_convergence(self):
    #    converged = False
    #    text = self.out.read_text().lower()
    #    if ('convergence targets achieved' in text and
    #        'reached self-consistent loops maximum' not in text):
    #        converged = True
    #    return converged

    #def read_number_of_electrons(self):
    #    nelec = None
    #    text = self.out.read_text().lower()
        # Total electronic charge
    #    for line in iter(text.split('\n')):
    #        if line.rfind('total electronic charge :') > -1:
    #            nelec = float(line.split(':')[1].strip())
    #            break
    #    return nelec

    #def read_number_of_iterations(self):
    #    niter = None
    #    lines = self.out.read_text().splitlines()
    #    for line in lines:
    #        if line.rfind(' Loop number : ') > -1:
    #            niter = int(line.split(':')[1].split()[0].strip())  # last iter
    #    return niter

    #def read_magnetic_moment(self):
    #    magmom = None
    #    lines = self.out.read_text().splitlines()
    #    for line in lines:
    #        if line.rfind('total moment                :') > -1:
    #            magmom = float(line.split(':')[1].strip())  # last iter
    #    return magmom

    #def read_electronic_temperature(self):
    #    width = None
    #    text = self.out.read_text().lower()
    #    for line in iter(text.split('\n')):
    #        if line.rfind('smearing width :') > -1:
    #            width = float(line.split(':')[1].strip())
    #            break
    #    return width

    def read_eigval(self):
        with (self.path / 'EIGVAL.OUT').open() as fd:
            yield from parse_elk_eigval(fd)


def parse_elk_kpoints(fd):
    header = next(fd)
    tokens = header.split()
    nkpts = int(tokens[0])
    assert tokens[1] == ':'

    kpts = np.empty(nkpts, 3)
    weights = np.empty(nkpts)

    for ikpt in range(nkpts):
        line = next(fd)
        tokens = line.split()
        kpts[ikpt] = tokens[1:4]
        weights[ikpt] = tokens[4]
    yield 'kpts', kpts
    yield 'weights', weights


def parse_elk_info(fd):
    #def skipto(regex):
    #    match = None
    #    while match is None:
    #        match = re.match(regex, line)
    #        line = next(fd)
    #    return line

    dct = {}
    fd = iter(fd)

    converged = False
    actually_did_not_converge = False
    # Legacy code kept track of both these things, which is strange.
    # Why could a file both claim to converge *and* not converge?

    current_indent = 0
    # We loop over all lines and extract also data that occurs
    # multiple times (e.g. in multiple SCF steps)
    for line in fd:
        # "name of quantity  :   1 2 3"
        match = re.match(r'\s*(\S+)\s+:\s+(.+)?\s*', line)
        if match is not None:
            key, values = match.group(1, 2)
            dct[key.strip()] = values

        elif line.startswith('Convergence targets achieved'):
            converged = True
        elif 'reached self-consistent loops maximum' in line.lower():
            actually_did_not_converge = True

    yield 'converged', converged and not actually_did_not_converge  # XXX
    yield 'charge', float(dct['Total electronic charge'])
    yield 'fermilevel', float(dct['Fermi'])


def parse_elk_eigval(fd):

    def match_int(line, word):
        number, colon, word1 = line.split()
        assert word1 == word
        assert colon == ':'
        return int(number)

    def skip_spaces(line=''):
        while not line.strip():
            line = next(fd)
        return line

    line = skip_spaces()
    nkpts = match_int(line, 'nkpt')  # 10 : nkpts
    line = next(fd)
    nbands = match_int(line, 'nstsv')  # 15 : nstsv
    # XXXX Spin polarized
    #
    #if self.get_spin_polarized():
    #    nbands = nbands // 2

    eigenvalues = np.empty((nkpts, nbands))
    occupations = np.empty((nkpts, nbands))
    kpts = np.empty((nkpts, 3))

    for ikpt in range(nkpts):
        line = skip_spaces()
        tokens = line.split()
        assert tokens[-1] == 'vkl', tokens
        assert ikpt + 1 == int(tokens[0])
        kpts[ikpt] = np.array(tokens[1:4]).astype(float)

        line = next(fd)  # "(state, eigenvalue and occupancy below)"
        assert line.strip().startswith('(state,'), line
        for iband in range(nbands):
            line = next(fd)
            tokens = line.split()  # (band number, eigenval, occ)
            assert iband + 1 == int(tokens[0])
            eigenvalues[ikpt, iband] = float(tokens[1])
            occupations[ikpt, iband] = float(tokens[2])

    yield 'ibz_kpoints', kpts
    yield 'eigenvalues', eigenvalues[None]
    yield 'occupations', occupations[None]
