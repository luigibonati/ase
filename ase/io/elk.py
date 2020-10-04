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


class ElkReader:
    def __init__(self, path):
        self.path = Path(path)

    def read_everything(self):
        converged = self.read_convergence()
        if not converged:
            raise RuntimeError(f'ELK did not converge! Check {self.out}')
        yield from self.read_energy()
        #if self.parameters.get('tforce'):
        yield 'forces', self.read_forces()

        yield 'width', self.read_electronic_temperature()
        yield 'nbands', self.read_number_of_bands()
        yield 'nelect', self.read_number_of_electrons()
        yield 'niter', self.read_number_of_iterations()
        yield 'magnetic_moment', self.read_magnetic_moment()

    @property
    def out(self):
        return self.path / 'INFO.OUT'

    def read_energy(self):
        txt = (self.path / 'TOTENERGY.OUT').read_text()
        tokens = txt.split()
        e = float(tokens[-1]) * Hartree
        yield 'free_energy', e
        yield 'energy', e

    def read_forces(self):
        lines = self.out.read_text().splitlines()
        forces = []
        for line in lines:
            if line.rfind('total force') > -1:
                forces.append(np.array([float(f)
                                        for f in line.split(':')[1].split()]))
        return np.array(forces) * Hartree / Bohr

    def read_convergence(self):
        converged = False
        text = self.out.read_text().lower()
        if ('convergence targets achieved' in text and
            'reached self-consistent loops maximum' not in text):
            converged = True
        return converged

    def read_kpts(self, mode='ibz_k_points'):
        """ Returns list of kpts weights or kpts coordinates.  """
        values = []
        assert mode in ['ibz_k_points', 'k_point_weights']

        lines = (self.path / 'KPOINTS.OUT').read_text().splitlines()

        kpts = None
        for line in lines:
            if line.rfind(': nkpt') > -1:
                kpts = int(line.split(':')[0].strip())
                break
        assert kpts is not None
        text = lines[1:]  # remove first line
        values = []
        for line in text:
            if mode == 'ibz_k_points':
                b = [float(c.strip()) for c in line.split()[1:4]]
            else:
                b = float(line.split()[-2])
            values.append(b)
        if len(values) == 0:
            values = None
        return np.array(values)

    def read_number_of_electrons(self):
        nelec = None
        text = self.out.read_text().lower()
        # Total electronic charge
        for line in iter(text.split('\n')):
            if line.rfind('total electronic charge :') > -1:
                nelec = float(line.split(':')[1].strip())
                break
        return nelec

    def read_number_of_iterations(self):
        niter = None
        lines = self.out.read_text().splitlines()
        for line in lines:
            if line.rfind(' Loop number : ') > -1:
                niter = int(line.split(':')[1].split()[0].strip())  # last iter
        return niter

    def read_magnetic_moment(self):
        magmom = None
        lines = self.out.read_text().splitlines()
        for line in lines:
            if line.rfind('total moment                :') > -1:
                magmom = float(line.split(':')[1].strip())  # last iter
        return magmom

    def read_electronic_temperature(self):
        width = None
        text = self.out.read_text().lower()
        for line in iter(text.split('\n')):
            if line.rfind('smearing width :') > -1:
                width = float(line.split(':')[1].strip())
                break
        return width

    def read_number_of_bands(self):
        nbands = None
        lines = (self.path / 'EIGVAL.OUT').read_text().splitlines()
        for line in lines:
            if line.rfind(': nstsv') > -1:
                nbands = int(line.split(':')[0].strip())
                break

        # XXXX Spin polarized
        #
        #if self.get_spin_polarized():
        #    nbands = nbands // 2
        return nbands

    def read_eigenvalues(self, kpt=0, spin=0, mode='eigenvalues'):
        """ Returns list of last eigenvalues, occupations
        for given kpt and spin.  """
        values = []
        assert mode in ['eigenvalues', 'occupations']
        lines = (self.path / 'EIGVAL.OUT').read_text().splitlines()
        nstsv = None
        for line in lines:
            if line.rfind(': nstsv') > -1:
                nstsv = int(line.split(':')[0].strip())
                break
        assert nstsv is not None
        kpts = None
        for line in lines:
            if line.rfind(': nkpt') > -1:
                kpts = int(line.split(':')[0].strip())
                break
        assert kpts is not None
        text = lines[3:]  # remove first 3 lines
        # find the requested k-point
        beg = 2 + (nstsv + 4) * kpt
        end = beg + nstsv
        if self.get_spin_polarized():
            # elk prints spin-up and spin-down together
            if spin == 0:
                beg = beg
                end = beg + nstsv // 2
            else:
                beg = beg + nstsv // 2
                end = end
        values = []
        for line in text[beg:end]:
            b = [float(c.strip()) for c in line.split()[1:]]
            values.append(b)
        if mode == 'eigenvalues':
            values = [Hartree * v[0] for v in values]
        else:
            values = [v[1] for v in values]
        if len(values) == 0:
            values = None
        return np.array(values)

    def read_fermi(self):
        """Method that reads Fermi energy in Hartree from the output file
        and returns it in eV"""
        E_f = None
        text = self.out.read_text().lower()
        for line in iter(text.split('\n')):
            if line.rfind('fermi                       :') > -1:
                E_f = float(line.split(':')[1].strip())
        E_f = E_f * Hartree
        return E_f
