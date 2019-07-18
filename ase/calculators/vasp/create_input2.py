# Copyright (C) 2008 CSC - Scientific Computing Ltd.
"""This module defines an ASE interface to VASP.

Developed on the basis of modules by Jussi Enkovaara and John
Kitchin.  The path of the directory containing the pseudopotential
directories (potpaw,potpaw_GGA, potpaw_PBE, ...) should be set
by the environmental flag $VASP_PP_PATH.

The user should also set the environmental flag $VASP_SCRIPT pointing
to a python script looking something like::

   import os
   exitcode = os.system('vasp')

Alternatively, user can set the environmental flag $VASP_COMMAND pointing
to the command use the launch vasp e.g. 'vasp' or 'mpirun -n 16 vasp'

http://cms.mpi.univie.ac.at/vasp/
"""

import os
import sys
import warnings
import shutil
from os.path import join, isfile, islink

import numpy as np

from ase.calculators.calculator import kpts2ndarray
from ase.utils import basestring

from ase.calculators.vasp.setups import setups_defaults

# Special keys we want to write in scientific notation
exp_keys = [
    'ediff',      # stopping-criterion for electronic upd.
    'ediffg',     # stopping-criterion for ionic upd.
    'symprec',    # precession in symmetry routines
    # The next keywords pertain to the VTST add-ons from Graeme Henkelman's
    # group at UT Austin
    'fdstep',     # Finite diference step for IOPT = 1 or 2
]

class GenerateVaspInput:
    xc_defaults = {
        'lda': {'pp': 'LDA'},
        # GGAs
        'pw91': {'pp': 'PW91', 'gga': '91'},
        'pbe': {'pp': 'PBE', 'gga': 'PE'},
        'pbesol': {'gga': 'PS'},
        'revpbe': {'gga': 'RE'},
        'rpbe': {'gga': 'RP'},
        'am05': {'gga': 'AM'},
        # Meta-GGAs
        'tpss': {'metagga': 'TPSS'},
        'revtpss': {'metagga': 'RTPSS'},
        'm06l': {'metagga': 'M06L'},
        'ms0': {'metagga': 'MS0'},
        'ms1': {'metagga': 'MS1'},
        'ms2': {'metagga': 'MS2'},
        'scan': {'metagga': 'SCAN'},
        'scan-rvv10': {'metagga': 'SCAN', 'luse_vdw': True, 'bparam': 15.7},
        # vdW-DFs
        'vdw-df': {'gga': 'RE', 'luse_vdw': True, 'aggac': 0.},
        'optpbe-vdw': {'gga': 'OR', 'luse_vdw': True, 'aggac': 0.0},
        'optb88-vdw': {'gga': 'BO', 'luse_vdw': True, 'aggac': 0.0,
                       'param1': 1.1 / 6.0, 'param2': 0.22},
        'optb86b-vdw': {'gga': 'MK', 'luse_vdw': True, 'aggac': 0.0,
                        'param1': 0.1234, 'param2': 1.0},
        'vdw-df2': {'gga': 'ML', 'luse_vdw': True, 'aggac': 0.0,
                    'zab_vdw': -1.8867},
        'rev-vdw-df2': {'gga': 'MK', 'luse_vdw': True, 'param1': 0.1234,
                        'param2':0.711357, 'zab_vdw': -1.8867, 'aggac': 0.0},
        'beef-vdw': {'gga': 'BF', 'luse_vdw': True,
                     'zab_vdw': -1.8867},
        # Hartree-Fock and hybrids
        'hf': {'lhfcalc': True, 'aexx': 1.0, 'aldac': 0.0,
               'aggac': 0.0},
        'b3lyp': {'gga': 'B3', 'lhfcalc': True, 'aexx': 0.2,
                  'aggax': 0.72, 'aggac': 0.81, 'aldac': 0.19},
        'pbe0': {'gga': 'PE', 'lhfcalc': True},
        'hse03': {'gga': 'PE', 'lhfcalc': True, 'hfscreen': 0.3},
        'hse06': {'gga': 'PE', 'lhfcalc': True, 'hfscreen': 0.2},
        'hsesol': {'gga': 'PS', 'lhfcalc': True, 'hfscreen': 0.2}
    }

    def __init__(self):
        self.ase_params = {
            'xc': None,  # Exchange-correlation recipe (e.g. 'B3LYP')
            'pp': None,  # Pseudopotential file (e.g. 'PW91')
            'setups': None,  # Special setups (e.g pv, sv, ...)
            'txt': '-',  # Where to send information
            # number of points between points in band structures:
            'kpts_nintersections': None,
            # Option to write explicit k-points in units
            # of reciprocal lattice vectors:
            'reciprocal': False,
            'kpts': (1, 1, 1),  # k-points
            # Option to use gamma-sampling instead of Monkhorst-Pack:
            'gamma': False,
            # Switch to disable writing constraints to POSCAR
            'ignore_constraints': False,
            # Net charge for the whole system; determines nelect if not 0
            'net_charge': None,
            'ldau_luj': None,
        }
        self.input_params = self.ase_params  # For compatibility

        # Default settings for INCAR
        self.vasp_params = {}


    def set_xc_params(self, xc):
        """Set parameters corresponding to XC functional"""
        xc = xc.lower()
        if xc is None:
            pass
        elif xc not in self.xc_defaults:
            xc_allowed = ', '.join(self.xc_defaults.keys())
            raise ValueError(
                '{0} is not supported for xc! Supported xc values'
                'are: {1}'.format(xc, xc_allowed))
        else:
            # XC defaults to PBE pseudopotentials
            if 'pp' not in self.xc_defaults[xc]:
                self.set(pp='pbe')
            self.set(**self.xc_defaults[xc])


    def set(self, **kwargs):
        for key, value in kwargs.items():
            key = key.lower()
            if key == 'xc':
                self.set_xc_params(value)

            if key in self.ase_params.keys():
                self.ase_params[key] = value
            else:
                self.vasp_params[key] = value

        # Check for key conflicts
        self.check_special_keys()

    def check_xc(self):
        """Make sure the calculator has functional & pseudopotentials set up

        If no XC combination, GGA functional or POTCAR type is specified,
        default to PW91. Otherwise, try to guess the desired pseudopotentials.
        """

        p = self.ase_params

        # There is no way to correctly guess the desired
        # set of pseudopotentials without 'pp' being set.
        # Usually, 'pp' will be set by 'xc'.
        if 'pp' not in p or p['pp'] is None:
            gga = self.get('gga', None)
            if gga is None:
                p.update({'pp': 'lda'})
            elif gga == '91':
                p.update({'pp': 'pw91'})
            elif gga == 'PE':
                p.update({'pp': 'pbe'})
            else:
                raise NotImplementedError(
                    "Unable to guess the desired set of pseudopotential"
                    "(POTCAR) files. Please do one of the following: \n"
                    "1. Use the 'xc' parameter to define your XC functional."
                    "These 'recipes' determine the pseudopotential file as "
                    "well as setting the INCAR parameters.\n"
                    "2. Use the 'gga' settings None (default), 'PE' or '91'; "
                    "these correspond to LDA, PBE and PW91 respectively.\n"
                    "3. Set the POTCAR explicitly with the 'pp' flag. The "
                    "value should be the name of a folder on the VASP_PP_PATH"
                    ", and the aliases 'LDA', 'PBE' and 'PW91' are also"
                    "accepted.\n")

        if (p['xc'] is not None and
                p['xc'].lower() == 'lda' and
                p['pp'].lower() != 'lda'):
            warnings.warn("XC is set to LDA, but PP is set to "
                          "{0}. \nThis calculation is using the {0} "
                          "POTCAR set. \n Please check that this is "
                          "really what you intended!"
                          "\n".format(p['pp'].upper()))


    def get(self, key, default=None):
        if key in self.ase_params:
            return self.ase_params[key]
        return self.vasp_params.get(key, default)


    def check_special_keys(self):
        # Check LDAU settings
        if self.get('ldau_luj') is not None:
            if (self.get('ldauu') is not None or
                self.get('ldaul') is not None or
                self.get('ldauj') is not None):
                raise NotImplementedError(
                    'You can either specify ldaul, ldauu, and ldauj OR '
                    'ldau_luj. ldau_luj is not a VASP keyword. It is a '
                    'dictionary that specifies L, U and J for each '
                    'chemical species in the atoms object. '
                    'For example fora water molecule:'
                    '''ldau_luj={'H':{'L':2, 'U':4.0, 'J':0.9},
                    'O':{'L':2, 'U':4.0, 'J':0.9}}''')

        # Check net_charge and nelect
        if (self.get('net_charge') is not None and
            self.get('nelect') is not None):
            raise NotImplementedError('Cannot specify both "net_charge"'
                                      ' and "nelect"')



    def initialize(self, atoms):
        """Initialize a VASP calculation

        Constructs the POTCAR file (does not actually write it).
        User should specify the PATH
        to the pseudopotentials in VASP_PP_PATH environment variable

        The pseudopotentials are expected to be in:
        LDA:  $VASP_PP_PATH/potpaw/
        PBE:  $VASP_PP_PATH/potpaw_PBE/
        PW91: $VASP_PP_PATH/potpaw_GGA/

        if your pseudopotentials are somewhere else, or named
        differently you may make symlinks at the paths above that
        point to the right place. Alternatively, you may pass the full
        name of a folder on the VASP_PP_PATH to the 'pp' parameter.
        """

        p = self.ase_params

        self.check_xc()
        self.all_symbols = atoms.get_chemical_symbols()
        self.natoms = len(atoms)

        self.spinpol = (atoms.get_initial_magnetic_moments().any()
                        or self.get('ispin') == 2)
        atomtypes = atoms.get_chemical_symbols()

        # Determine the number of atoms of each atomic species
        # sorted after atomic species
        special_setups = []
        symbols = []
        symbolcount = {}

        # Default setup lists are available: 'minimal', 'recommended' and 'GW'
        # These may be provided as a string e.g.::
        #
        #     calc = Vasp(setups='recommended')
        #
        # or in a dict with other specifications e.g.::
        #
        #    calc = Vasp(setups={'base': 'minimal', 'Ca': '_sv', 2: 'O_s'})
        #
        # Where other keys are either atom identities or indices, and the
        # corresponding values are suffixes or the full name of the setup
        # folder, respectively.

        # Default to minimal basis
        if p['setups'] is None:
            p['setups'] = {'base': 'minimal'}

        # String shortcuts are initialised to dict form
        elif isinstance(p['setups'], str):
            if p['setups'].lower() in setups_defaults.keys():
                p['setups'] = {'base': p['setups']}

        # Dict form is then queried to add defaults from setups.py.
        if 'base' in p['setups']:
            setups = setups_defaults[p['setups']['base'].lower()]
        else:
            setups = {}

        # Override defaults with user-defined setups
        if p['setups'] is not None:
            setups.update(p['setups'])

        for m in setups:
            try:
                special_setups.append(int(m))
            except ValueError:
                continue

        for m, atom in enumerate(atoms):
            symbol = atom.symbol
            if m in special_setups:
                pass
            else:
                if symbol not in symbols:
                    symbols.append(symbol)
                    symbolcount[symbol] = 1
                else:
                    symbolcount[symbol] += 1

        # Build the sorting list
        self.sort = []
        self.sort.extend(special_setups)

        for symbol in symbols:
            for m, atom in enumerate(atoms):
                if m in special_setups:
                    pass
                else:
                    if atom.symbol == symbol:
                        self.sort.append(m)
        self.resort = list(range(len(self.sort)))
        for n in range(len(self.resort)):
            self.resort[self.sort[n]] = n
        self.atoms_sorted = atoms[self.sort]

        # Check if the necessary POTCAR files exists and
        # create a list of their paths.
        self.symbol_count = []
        for m in special_setups:
            self.symbol_count.append([atomtypes[m], 1])
        for m in symbols:
            self.symbol_count.append([m, symbolcount[m]])

        sys.stdout.flush()

        # Potpaw folders may be identified by an alias or full name
        for pp_alias, pp_folder in (('lda', 'potpaw'),
                                    ('pw91', 'potpaw_GGA'),
                                    ('pbe', 'potpaw_PBE')):
            if p['pp'].lower() == pp_alias:
                break
        else:
            pp_folder = p['pp']

        if 'VASP_PP_PATH' in os.environ:
            pppaths = os.environ['VASP_PP_PATH'].split(':')
        else:
            pppaths = []
        self.ppp_list = []
        # Setting the pseudopotentials, first special setups and
        # then according to symbols
        for m in special_setups:
            if m in setups:
                special_setup_index = m
            elif str(m) in setups:
                special_setup_index = str(m)
            else:
                raise Exception("Having trouble with special setup index {0}."
                                " Please use an int.".format(m))
            potcar = join(pp_folder,
                          setups[special_setup_index],
                          'POTCAR')
            for path in pppaths:
                filename = join(path, potcar)

                if isfile(filename) or islink(filename):
                    self.ppp_list.append(filename)
                    break
                elif isfile(filename + '.Z') or islink(filename + '.Z'):
                    self.ppp_list.append(filename + '.Z')
                    break
            else:
                print('Looking for %s' % potcar)
                raise RuntimeError('No pseudopotential for %s!' % symbol)

        for symbol in symbols:
            try:
                potcar = join(pp_folder, symbol + setups[symbol],
                              'POTCAR')
            except (TypeError, KeyError):
                potcar = join(pp_folder, symbol, 'POTCAR')
            for path in pppaths:
                filename = join(path, potcar)

                if isfile(filename) or islink(filename):
                    self.ppp_list.append(filename)
                    break
                elif isfile(filename + '.Z') or islink(filename + '.Z'):
                    self.ppp_list.append(filename + '.Z')
                    break
            else:
                print('''Looking for %s
                The pseudopotentials are expected to be in:
                LDA:  $VASP_PP_PATH/potpaw/
                PBE:  $VASP_PP_PATH/potpaw_PBE/
                PW91: $VASP_PP_PATH/potpaw_GGA/''' % potcar)
                raise RuntimeError('No pseudopotential for %s!' % symbol)

        self.converged = None
        self.setups_changed = None


    def default_nelect_from_ppp(self):
        """ Get default number of electrons from ppp_list and symbol_count

        "Default" here means that the resulting cell would be neutral.
        """
        symbol_valences = []
        for filename in self.ppp_list:
            ppp_file = open_potcar(filename=filename)
            r = read_potcar_numbers_of_electrons(ppp_file)
            symbol_valences.extend(r)
            ppp_file.close()
        assert len(self.symbol_count) == len(symbol_valences)
        default_nelect = 0
        for ((symbol1, count), (symbol2, valence)) in zip(self.symbol_count,
                                                          symbol_valences):
            assert symbol1 == symbol2
            default_nelect += count * valence
        return default_nelect


    def write_input(self, atoms, directory='./'):
        from ase.io.vasp import write_vasp
        write_vasp(join(directory, 'POSCAR'),
                   self.atoms_sorted,
                   symbol_count=self.symbol_count,
                   ignore_constraints=self.get('ignore_constraints'))
        self.write_incar(atoms, directory=directory)
        self.write_potcar(directory=directory)
        self.write_kpoints(directory=directory)
        self.write_sort_file(directory=directory)
        self.copy_vdw_kernel(directory=directory)


    def copy_vdw_kernel(self, directory='./'):
        """Method to copy the vdw_kernel.bindat file.
        Set ASE_VASP_VDW environment variable to the vdw_kernel.bindat
        folder location. Checks if LUSE_VDW is enabled, and if no location
        for the vdW kernel is specified, a warning is issued."""

        vdw_env = 'ASE_VASP_VDW'
        kernel = 'vdw_kernel.bindat'
        dst = os.path.join(directory, kernel)

        # No need to copy the file again
        if isfile(dst):
            return

        if self.get('luse_vdw'):
            src = None
            if vdw_env in os.environ:
                src = os.path.join(os.environ[vdw_env],
                                   kernel)

            if not src or not isfile(src):
                warnings.warn(('vdW has been enabled, however no'
                               ' location for the {} file'
                               ' has been specified.'
                               ' Set {} environment variable to'
                               ' copy the vdW kernel.').format(
                                   kernel, vdw_env))
            else:
                shutil.copyfile(src, dst)


    def clean(self, directory='./'):
        """Method which cleans up after a calculation.

        The default files generated by Vasp will be deleted IF this
        method is called.

        """
        files = ['CHG', 'CHGCAR', 'POSCAR', 'INCAR', 'CONTCAR',
                 'DOSCAR', 'EIGENVAL', 'IBZKPT', 'KPOINTS', 'OSZICAR',
                 'OUTCAR', 'PCDAT', 'POTCAR', 'vasprun.xml',
                 'WAVECAR', 'XDATCAR', 'PROCAR', 'ase-sort.dat',
                 'LOCPOT', 'AECCAR0', 'AECCAR1', 'AECCAR2']
        for f in files:
            try:
                os.remove(os.path.join(directory, f))
            except OSError:
                pass

    def _format_single(self, n):
        if isinstance(n, str):
            return n
        elif isinstance(n, bool):
            # Handle bool before int!
            # PEP 285
            if n is True:
                return '.TRUE.'
            else:
                return '.FALSE.'
        elif isinstance(n, int):
            return '{:d}'.format(n)
        elif isinstance(n, float):
            return '{:.4f}'.format(n)
        else:
            # What could go here?
            return format('{}'.format(n))

    def write_incar(self, atoms, directory='./', **kwargs):

        self.check_special_keys()  # Ensure consistency

        # Create new dict with {key: str} type entries, which will be written
        # at the end
        inputs = {}

        # Parse special keys
        val = self.get('ldau_luj')
        if val:
            if self.get('ldau') is None:
                # We will write this later
                self.set(ldau=True)
            llist = ulist = jlist = ''
            for symbol in self.symbol_count:
                # Default: No +U
                luj = val.get(symbol[0], {'L': -1, 'U': 0.0, 'J': 0.0})
                llist += ' {:d}'.format(luj['L'])
                ulist += ' {:.3f}'.format(luj['U'])
                jlist += ' {:.3f}'.format(luj['J'])
            inputs['ldaul'] = llist
            inputs['ldauu'] = ulist
            inputs['ldauj'] = jlist

        # Parse magmoms
        # magmoms keyword takes precedence over initial_magnetic_moments()
        if self.get('magmom') is None and atoms.get_initial_magnetic_moments().any():
            if self.get('ispin') is None:
                # We will write this later
                self.set(ispin=2)
            # Write out initial magnetic moments
            magmom = atoms.get_initial_magnetic_moments()[self.sort]
            # unpack magmom array if three components specified
            if magmom.ndim > 1:
                magmom = [item for sublist in magmom for item in sublist]
            lst = [[1, magmom[0]]]
            for n in range(1, len(magmom)):
                if magmom[n] == magmom[n - 1]:
                    lst[-1][0] += 1
                else:
                    lst.append([1, magmom[n]])

            lst = ' '.join('{:d}*{:.4f}'.format(*mom) for mom in lst)
            inputs['magmom'] = lst

        # Parse net_charge
        val = self.get('net_charge')
        if val is not None and val != 0:
            # We already checked net_charge and nelect aren't
            # both defined
            default_nelect = self.default_nelect_from_ppp()
            inputs['nelect'] = default_nelect + val

        # Parse remaining settings
        for key, val in self.vasp_params.items():
            if isinstance(val, basestring):
                inputs[key] = val
            elif key in exp_keys:
                # Special case for a few keys, that we want to write in exponential format
                inputs[key] = '{:.2e}'.format(val)
            elif isinstance(val, (list, tuple, np.ndarray)):
                # List of stuff. Does not work with list of lists right now
                s = ' '.join(self._format_single(n) for n in val)
                inputs[key] =  s
            else:
                s = self._format_single(val)
                inputs[key] = s

        # Write stuff in inputs into INCAR file
        with open(os.path.join(directory, 'INCAR'), 'w') as incar:
            print('INCAR created by Atomic Simulation Environment', file=incar)
            for key, value in sorted(inputs.items()):
                print(' {} = {}'.format(key.upper(), value), file=incar)
            incar.flush()


    def write_kpoints(self, directory='./', **kwargs):
        """Writes the KPOINTS file."""

        # Don't write anything if KSPACING is being used
        if self.get('kspacing') is not None:
            ksp = self.get('kspacing')
            if ksp > 0:
                return
            else:
                raise ValueError("KSPACING value {0} is not allowable. "
                                 "Please use None or a positive number."
                                 "".format(ksp))

        p = self.ase_params
        kpoints = open(join(directory, 'KPOINTS'), 'w')
        kpoints.write('KPOINTS created by Atomic Simulation Environment\n')

        if isinstance(p['kpts'], dict):
            p['kpts'] = kpts2ndarray(p['kpts'], atoms=self.atoms)
            p['reciprocal'] = True

        shape = np.array(p['kpts']).shape

        # Wrap scalar in list if necessary
        if shape == ():
            p['kpts'] = [p['kpts']]
            shape = (1, )

        if len(shape) == 1:
            kpoints.write('0\n')
            if shape == (1, ):
                kpoints.write('Auto\n')
            elif p['gamma']:
                kpoints.write('Gamma\n')
            else:
                kpoints.write('Monkhorst-Pack\n')
            [kpoints.write('%i ' % kpt) for kpt in p['kpts']]
            kpoints.write('\n0 0 0\n')
        elif len(shape) == 2:
            kpoints.write('%i \n' % (len(p['kpts'])))
            if p['reciprocal']:
                kpoints.write('Reciprocal\n')
            else:
                kpoints.write('Cartesian\n')
            for n in range(len(p['kpts'])):
                [kpoints.write('%f ' % kpt) for kpt in p['kpts'][n]]
                if shape[1] == 4:
                    kpoints.write('\n')
                elif shape[1] == 3:
                    kpoints.write('1.0 \n')
        kpoints.close()


    def write_potcar(self, suffix="", directory='./'):
        """Writes the POTCAR file."""
        potfile = open(join(directory, 'POTCAR' + suffix), 'w')
        for filename in self.ppp_list:
            ppp_file = open_potcar(filename=filename)
            for line in ppp_file:
                potfile.write(line)
            ppp_file.close()
        potfile.close()


    def write_sort_file(self, directory='./'):
        """Writes a sortings file.

        This file contains information about how the atoms are sorted in
        the first column and how they should be resorted in the second
        column. It is used for restart purposes to get sorting right
        when reading in an old calculation to ASE."""

        with open(join(directory, 'ase-sort.dat'), 'w') as f:
            for n in range(len(self.sort)):
                print('{:d} {:d}'.format(self.sort[n], self.resort[n]), file=f)


def open_potcar(filename):
    """ Open POTCAR file with transparent decompression if it's an archive (.Z)
    """
    import gzip
    if filename.endswith('R'):
        return open(filename, 'r')
    elif filename.endswith('.Z'):
        return gzip.open(filename)
    else:
        raise ValueError('Invalid POTCAR filename: "%s"' % filename)

def read_potcar_numbers_of_electrons(file_obj):
    """ Read list of tuples (atomic symbol, number of valence electrons)
    for each atomtype from a POTCAR file."""
    nelect = []
    lines = file_obj.readlines()
    for n, line in enumerate(lines):
        if 'TITEL' in line:
            symbol = line.split('=')[1].split()[1].split('_')[0].strip()
            valence = float(lines[n + 4].split(';')[1]
                            .split('=')[1].split()[0].strip())
            nelect.append((symbol, valence))
    return nelect
