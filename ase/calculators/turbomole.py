# type: ignore
"""
This module defines an ASE interface to Turbomole: http://www.turbomole.com/

QMMM functionality provided by Markus Kaukonen <markus.kaukonen@iki.fi>.

Please read the license file (../../LICENSE)

Contact: Ivan Kondov <ivan.kondov@kit.edu>
"""
import os
import re
import warnings
from math import log10, floor
import numpy as np
from ase.units import Ha, Bohr
from ase.io import read, write
from ase.calculators.calculator import FileIOCalculator
from ase.calculators.calculator import PropertyNotImplementedError, ReadError
from ase.calculators.turbomole_executor import execute
from ase.calculators.turbomole_writer import add_data_group, delete_data_group
from ase.calculators import turbomole_reader
from ase.calculators.turbomole_reader import read_data_group, read_convergence
from ase.calculators.turbomole_parameters import TurbomoleParameters


class TurbomoleOptimizer:
    def __init__(self, atoms, calc):
        self.atoms = atoms
        self.calc = calc
        self.atoms.calc = self.calc

    def todict(self):
        return {'type': 'optimization',
                'optimizer': 'TurbomoleOptimizer'}

    def run(self, fmax=None, steps=None):
        if fmax is not None:
            self.calc.parameters['force convergence'] = fmax
            self.calc.verify_parameters()
        if steps is not None:
            self.calc.parameters['geometry optimization iterations'] = steps
            self.calc.verify_parameters()
        self.calc.calculate()
        self.atoms.positions[:] = self.calc.atoms.positions
        self.calc.parameters['task'] = 'energy'


class Turbomole(FileIOCalculator):

    """constants"""
    name = 'Turbomole'

    implemented_properties = ['energy', 'forces', 'dipole', 'free_energy',
                              'charges']

    tm_files = [
        'control', 'coord', 'basis', 'auxbasis', 'energy', 'gradient', 'mos',
        'alpha', 'beta', 'statistics', 'GEO_OPT_CONVERGED', 'GEO_OPT_FAILED',
        'not.converged', 'nextstep', 'hessapprox', 'job.last', 'job.start',
        'optinfo', 'statistics', 'converged', 'vibspectrum',
        'vib_normal_modes', 'hessian', 'dipgrad', 'dscf_problem', 'pc.txt',
        'pc_gradients.txt'
    ]
    tm_tmp_files = [
        'errvec', 'fock', 'oldfock', 'dens', 'ddens', 'diff_densmat',
        'diff_dft_density', 'diff_dft_oper', 'diff_fockmat', 'diis_errvec',
        'diis_oldfock'
    ]

    # initialize attributes
    results = {}
    initialized = False
    pc_initialized = False
    converged = False
    updated = False
    update_energy = None
    update_forces = None
    update_geometry = None
    update_hessian = None
    atoms = None
    forces = None
    e_total = None
    dipole = None
    charges = None
    version = None
    runtime = None
    datetime = None
    hostname = None
    pcpot = None

    def __init__(self, label=None, calculate_energy='dscf',
                 calculate_forces='grad', post_HF=False, atoms=None,
                 restart=False, define_str=None, control_kdg=None,
                 control_input=None, reset_tolerance=1e-2, **kwargs):

        super().__init__()
        self.parameters = TurbomoleParameters()

        self.label = label
        self.calculate_energy = calculate_energy
        self.calculate_forces = calculate_forces
        self.post_HF = post_HF
        self.restart = restart
        self.define_str = define_str
        self.control_kdg = control_kdg
        self.control_input = control_input
        self.reset_tolerance = reset_tolerance

        if self.restart:
            self._set_restart(kwargs)
        else:
            self.parameters.update(kwargs)
            self.set_parameters()
            self.verify_parameters()
            self.reset()

        if atoms is not None:
            atoms.calc = self
            self.set_atoms(atoms)

    def __getitem__(self, item):
        return getattr(self, item)

    def _set_restart(self, params_update):
        """constructs atoms, parameters and results from a previous
        calculation"""

        # read results, key parameters and non-key parameters
        self.read_restart()
        params_old = self.parameters.read_parameters(self.atoms, self.results)

        # filter out non-updateable parameters
        for p in list(params_update.keys()):
            if not self.parameters.is_updateable(p):
                del params_update[p]
                warnings.warn('"' + p + '"' + ' cannot be changed')

        # update and verify parameters
        params_new = params_old.copy()
        params_new.update(params_update)
        self.parameters.update(params_new)
        self.set_parameters()
        self.verify_parameters()

        # if a define string is specified then run define
        if self.define_str:
            execute('define', input_str=self.define_str)

        # updates data groups in the control file
        if params_update or self.control_kdg or self.control_input:
            self.parameters.update_data_groups(params_old, params_update)
            self._set_post_define()

        self.initialized = True
        # more precise convergence tests are necessary to set these flags:
        self.update_energy = True
        self.update_forces = True
        self.update_geometry = True
        self.update_hessian = True

    def _set_post_define(self):
        """non-define keys, user-specified changes in the control file"""
        self.parameters.update_no_define_parameters()

        # delete user-specified data groups
        if self.control_kdg:
            for dg in self.control_kdg:
                delete_data_group(dg)

        # append user-defined input to control
        if self.control_input:
            for inp in self.control_input:
                add_data_group(inp, raw=True)

        # add point charges if pcpot defined:
        if self.pcpot:
            self.set_point_charges()

    def set_parameters(self):
        """loads the default parameters and updates with actual values"""
        if self.parameters.get('use resolution of identity'):
            self.calculate_energy = 'ridft'
            self.calculate_forces = 'rdgrad'

    def verify_parameters(self):
        if self.define_str:
            assert isinstance(self.define_str, str)
            assert len(self.define_str) != 0
        else:
            self.parameters.verify_parameters()

    def reset(self):
        """removes all turbomole input, output and scratch files,
        and deletes results dict and the atoms object"""
        self.atoms = None
        self.results = {}
        self.results['calculation parameters'] = {}
        ase_files = [f for f in os.listdir('.') if f.startswith('ASE.TM.')]
        for f in self.tm_files + self.tm_tmp_files + ase_files:
            if os.path.exists(f):
                os.remove(f)
        self.initialized = False
        self.pc_initialized = False
        self.converged = False

    def set_atoms(self, atoms):
        """Create the self.atoms object and writes the coord file. If
        self.atoms exists a check for changes and an update of the atoms
        is performed. Note: Only positions changes are tracked in this
        version.
        """
        changes = self.check_state(atoms, tol=1e-13)
        if self.atoms == atoms or 'positions' not in changes:
            # print('two atoms obj are (almost) equal')
            if self.updated and os.path.isfile('coord'):
                self.updated = False
                a = read('coord').get_positions()
                if np.allclose(a, atoms.get_positions(), rtol=0, atol=1e-13):
                    return
            else:
                return

        changes = self.check_state(atoms, tol=self.reset_tolerance)
        if 'positions' in changes:
            # print(two atoms obj are different')
            self.reset()
        else:
            # print('two atoms obj are slightly different')
            if self.parameters['use redundant internals']:
                self.reset()

        write('coord', atoms)
        self.atoms = atoms.copy()
        self.update_energy = True
        self.update_forces = True
        self.update_geometry = True
        self.update_hessian = True


    def initialize(self):
        """prepare turbomole control file by running module 'define'"""
        if self.initialized:
            return
        self.verify_parameters()
        if not self.atoms:
            raise RuntimeError('atoms missing during initialization')
        if not os.path.isfile('coord'):
            raise IOError('file coord not found')

        if self.define_str is not None:
            define_str = self.define_str
        else:
            define_str = self.parameters.get_define_str(len(self.atoms))

        # run define
        execute('define', input_str=define_str)

        # process non-default initial guess
        iguess = self.parameters['initial guess']
        if isinstance(iguess, dict) and 'use' in iguess.keys():
            # "use" initial guess
            if self.parameters['multiplicity'] != 1 or self.parameters['uhf']:
                define_str = '\n\n\ny\nuse ' + iguess['use'] + '\nn\nn\nq\n'
            else:
                define_str = '\n\n\ny\nuse ' + iguess['use'] + '\nn\nq\n'
            execute('define', input_str=define_str)
        elif self.parameters['initial guess'] == 'hcore':
            # "hcore" initial guess
            if self.parameters['multiplicity'] != 1 or self.parameters['uhf']:
                delete_data_group('uhfmo_alpha')
                delete_data_group('uhfmo_beta')
                add_data_group('uhfmo_alpha', 'none file=alpha')
                add_data_group('uhfmo_beta', 'none file=beta')
            else:
                delete_data_group('scfmo')
                add_data_group('scfmo', 'none file=mos')

        self._set_post_define()

        self.initialized = True
        self.converged = False

    def calculation_required(self, atoms, properties):
        if self.atoms != atoms:
            return True
        for prop in properties:
            if prop == 'energy' and self.e_total is None:
                return True
            elif prop == 'forces' and self.forces is None:
                return True
        return False

    def calculate(self, atoms=None):
        """execute the requested job"""
        if atoms is None:
            atoms = self.atoms
        if self.parameters['task'] in ['energy', 'energy calculation']:
            self.get_potential_energy(atoms)
        if self.parameters['task'] in ['gradient', 'gradient calculation']:
            self.get_forces(atoms)
        if self.parameters['task'] in ['optimize', 'geometry optimization']:
            self.relax_geometry(atoms)
        if self.parameters['task'] in ['frequencies', 'normal mode analysis']:
            self.normal_mode_analysis(atoms)
        self.read_results()

    def relax_geometry(self, atoms=None):
        """execute geometry optimization with script jobex"""
        if atoms is None:
            atoms = self.atoms
        self.set_atoms(atoms)
        if self.converged and not self.update_geometry:
            return
        self.initialize()
        jobex_flags = ''
        if self.parameters['use resolution of identity']:
            jobex_flags += ' -ri'
        if self.parameters['force convergence']:
            par = self.parameters['force convergence']
            conv = floor(-log10(par / Ha * Bohr))
            jobex_flags += ' -gcart ' + str(int(conv))
        if self.parameters['energy convergence']:
            par = self.parameters['energy convergence']
            conv = floor(-log10(par / Ha))
            jobex_flags += ' -energy ' + str(int(conv))
        geom_iter = self.parameters['geometry optimization iterations']
        if geom_iter is not None:
            assert isinstance(geom_iter, int)
            jobex_flags += ' -c ' + str(geom_iter)
        self.converged = False
        execute('jobex' + jobex_flags)
        # check convergence
        self.converged = read_convergence(self.restart, self.parameters)
        if self.converged:
            self.update_energy = False
            self.update_forces = False
            self.update_geometry = False
            self.update_hessian = True
        # read results
        new_struct = read('coord')
        atoms.set_positions(new_struct.get_positions())
        self.atoms = atoms.copy()
        self.read_energy()

    def normal_mode_analysis(self, atoms=None):
        """execute normal mode analysis with modules aoforce or NumForce"""
        from ase.constraints import FixAtoms
        if atoms is None:
            atoms = self.atoms
        self.set_atoms(atoms)
        self.initialize()
        if self.update_energy:
            self.get_potential_energy(atoms)
        if self.update_hessian:
            fixatoms = []
            for constr in atoms.constraints:
                if isinstance(constr, FixAtoms):
                    ckwargs = constr.todict()['kwargs']
                    if 'indices' in ckwargs.keys():
                        fixatoms.extend(ckwargs['indices'])
            if self.parameters['numerical hessian'] is None:
                if len(fixatoms) > 0:
                    define_str = '\n\ny\n'
                    for index in fixatoms:
                        define_str += 'm ' + str(index + 1) + ' 999.99999999\n'
                    define_str += '*\n*\nn\nq\n'
                    execute('define', input_str=define_str)
                    dg = read_data_group('atoms')
                    regex = r'(mass\s*=\s*)999.99999999'
                    dg = re.sub(regex, r'\g<1>9999999999.9', dg)
                    dg += '\n'
                    delete_data_group('atoms')
                    add_data_group(dg, raw=True)
                execute('aoforce')
            else:
                optstr = ''
                pdict = self.parameters['numerical hessian']
                if self.parameters['use resolution of identity']:
                    optstr += ' -ri'
                if len(fixatoms) > 0:
                    optstr += ' -frznuclei -central -c'
                if 'central' in pdict.keys():
                    optstr += ' -central'
                if 'delta' in pdict.keys():
                    optstr += ' -d ' + str(pdict['delta'] / Bohr)
                execute('NumForce' + optstr)
            self.update_hessian = False

    def read_restart(self):
        """read a previous calculation from control file"""
        self.atoms = read('coord')
        self.atoms.calc = self
        self.converged = read_convergence(self.restart, self.parameters)
        self.read_results()

    def read_results(self):
        """read all results and load them in the results entity"""
        read_methods = [
            self.read_energy,
            self.read_gradient,
            self.read_forces,
            self.read_basis_set,
            self.read_ecps,
            self.read_mos,
            self.read_occupation_numbers,
            self.read_dipole_moment,
            self.read_ssquare,
            self.read_hessian,
            self.read_vibrational_reduced_masses,
            self.read_normal_modes,
            self.read_vibrational_spectrum,
            self.read_charges,
            self.read_point_charges,
            self.read_run_parameters
        ]
        for method in read_methods:
            try:
                method()
            except ReadError as err:
                warnings.warn(err.args[0])

    def read_run_parameters(self):
        """read parameters set by define and not in self.parameters"""
        turbomole_reader.read_run_parameters(self.results)

    def read_energy(self):
        """Read energy from Turbomole energy file."""
        turbomole_reader.read_energy(self.results, self.post_HF)
        self.e_total = self.results['total energy']

    def read_forces(self):
        """Read forces from Turbomole gradient file."""
        self.forces = turbomole_reader.read_forces(self.results, len(self.atoms))

    def read_occupation_numbers(self):
        """read occupation numbers"""
        turbomole_reader.read_occupation_numbers(self.results)

    def read_mos(self):
        """read the molecular orbital coefficients and orbital energies
        from files mos, alpha and beta"""

        ans = turbomole_reader.read_mos(self.results)
        if ans is not None:
            self.converged = ans

    def read_basis_set(self):
        """read the basis set"""
        turbomole_reader.read_basis_set(self.results)

    def read_ecps(self):
        """read the effective core potentials"""
        turbomole_reader.read_ecps(self.results)

    def read_gradient(self):
        """read all information in file 'gradient'"""
        turbomole_reader.read_gradient(self.results)

    def read_hessian(self):
        """Read in the hessian matrix"""
        turbomole_reader.read_hessian(self.results)

    def read_normal_modes(self):
        """Read in vibrational normal modes"""
        turbomole_reader.read_normal_modes(self.results)

    def read_vibrational_reduced_masses(self):
        """Read vibrational reduced masses"""
        turbomole_reader.read_vibrational_reduced_masses(self.results)

    def read_vibrational_spectrum(self):
        """Read the vibrational spectrum"""
        turbomole_reader.read_vibrational_spectrum(self.results)

    def read_ssquare(self):
        """Read the expectation value of S^2 operator"""
        turbomole_reader.read_ssquare(self.results)

    def read_dipole_moment(self):
        """Read the dipole moment"""
        turbomole_reader.read_dipole_moment(self.results)
        dip_vec = self.results['electric dipole moment']['vector']['array']
        self.dipole = np.array(dip_vec) * Bohr

    def read_charges(self):
        """read partial charges on atoms from an ESP fit"""
        filename = 'ASE.TM.' + self.calculate_energy + '.out'
        self.charges = turbomole_reader.read_charges(filename, len(self.atoms))

    def get_version(self):
        """get the version of the installed turbomole package"""
        if not self.version:
            self.version = turbomole_reader.read_version()
        return self.version

    def get_datetime(self):
        """get the timestamp of most recent calculation"""
        if not self.datetime:
            self.datetime = turbomole_reader.read_datetime()
        return self.datetime

    def get_runtime(self):
        """get the total runtime of calculations"""
        if not self.runtime:
            self.runtime = turbomole_reader.read_runtime()
        return self.runtime

    def get_hostname(self):
        """get the hostname of the computer on which the calc has run"""
        if not self.hostname:
            self.hostname = turbomole_reader.read_hostname()
        return self.hostname

    def get_optimizer(self, atoms, trajectory=None, logfile=None):
        """returns a TurbomoleOptimizer object"""
        self.parameters['task'] = 'optimize'
        self.verify_parameters()
        return TurbomoleOptimizer(atoms, self)

    def get_results(self):
        """returns the results dictionary"""
        return self.results

    def get_potential_energy(self, atoms, force_consistent=True):
        # update atoms
        self.updated = self.e_total is None
        self.set_atoms(atoms)
        self.initialize()
        # if update of energy is necessary
        if self.update_energy:
            # calculate energy
            execute(self.calculate_energy)
            # check convergence
            self.converged = read_convergence(self.restart, self.parameters)
            if not self.converged:
                return None
            # read energy
            self.read_energy()

        self.update_energy = False
        return self.e_total

    def get_forces(self, atoms):
        # update atoms
        self.updated = self.forces is None
        self.set_atoms(atoms)
        # complete energy calculations
        if self.update_energy:
            self.get_potential_energy(atoms)
        # if update of forces is necessary
        if self.update_forces:
            # calculate forces
            execute(self.calculate_forces)
            # read forces
            self.read_forces()

        self.update_forces = False
        return self.forces.copy()

    def get_dipole_moment(self, atoms):
        self.get_potential_energy(atoms)
        self.read_dipole_moment()
        return self.dipole

    def get_property(self, name, atoms=None, allow_calculation=True):
        """return the value of a property"""

        if name not in self.implemented_properties:
            # an ugly work around; the caller should test the raised error
            # if name in ['magmom', 'magmoms', 'charges', 'stress']:
            # return None
            raise PropertyNotImplementedError(name)

        if atoms is None:
            atoms = self.atoms.copy()

        persist_property = {
            'energy': 'e_total',
            'forces': 'forces',
            'dipole': 'dipole',
            'free_energy': 'e_total',
            'charges': 'charges'
        }
        property_getter = {
            'energy': self.get_potential_energy,
            'forces': self.get_forces,
            'dipole': self.get_dipole_moment,
            'free_energy': self.get_potential_energy,
            'charges': self.get_charges
        }
        getter_args = {
            'energy': [atoms],
            'forces': [atoms],
            'dipole': [atoms],
            'free_energy': [atoms, True],
            'charges': [atoms]
        }

        if allow_calculation:
            result = property_getter[name](*getter_args[name])
        else:
            if hasattr(self, persist_property[name]):
                result = getattr(self, persist_property[name])
            else:
                result = None

        if isinstance(result, np.ndarray):
            result = result.copy()
        return result

    def get_charges(self, atoms):
        """return partial charges on atoms from an ESP fit"""
        self.get_potential_energy(atoms)
        self.read_charges()
        return self.charges

    def get_forces_on_point_charges(self):
        """return forces acting on point charges"""
        self.get_forces(self.atoms)
        lines = read_data_group('point_charge_gradients').split('\n')[1:]
        forces = []
        for line in lines:
            linef = line.strip().replace('D', 'E')
            forces.append([float(x) for x in linef.split()])
        # Note the '-' sign for turbomole, to get forces
        return -np.array(forces) * Ha / Bohr

    def set_point_charges(self, pcpot=None):
        """write external point charges to control"""
        if pcpot is not None and pcpot != self.pcpot:
            self.pcpot = pcpot
        if self.pcpot.mmcharges is None or self.pcpot.mmpositions is None:
            raise RuntimeError('external point charges not defined')

        if not self.pc_initialized:
            if len(read_data_group('point_charges')) == 0:
                add_data_group('point_charges', 'file=pc.txt')
            if len(read_data_group('point_charge_gradients')) == 0:
                add_data_group(
                    'point_charge_gradients',
                    'file=pc_gradients.txt'
                )
            drvopt = read_data_group('drvopt')
            if 'point charges' not in drvopt:
                drvopt += '\n   point charges\n'
                delete_data_group('drvopt')
                add_data_group(drvopt, raw=True)
            self.pc_initialized = True

        if self.pcpot.updated:
            with open('pc.txt', 'w') as pcfile:
                pcfile.write('$point_charges nocheck list\n')
                for (x, y, z), charge in zip(
                        self.pcpot.mmpositions, self.pcpot.mmcharges):
                    pcfile.write('%20.14f  %20.14f  %20.14f  %20.14f\n'
                                 % (x / Bohr, y / Bohr, z / Bohr, charge))
                pcfile.write('$end \n')
            self.pcpot.updated = False

    def read_point_charges(self):
        """read point charges from previous calculation"""
        charges, positions = turbomole_reader.read_point_charges()
        if len(charges) > 0:
            self.pcpot = PointChargePotential(charges, positions)

    def embed(self, charges=None, positions=None):
        """embed atoms in an array of point-charges; function used in
            qmmm calculations."""
        self.pcpot = PointChargePotential(charges, positions)
        return self.pcpot


class PointChargePotential:
    """Point-charge potential for Turbomole"""
    def __init__(self, mmcharges, mmpositions=None):
        self.mmcharges = mmcharges
        self.mmpositions = mmpositions
        self.mmforces = None
        self.updated = True

    def set_positions(self, mmpositions):
        """set the positions of point charges"""
        self.mmpositions = mmpositions
        self.updated = True

    def set_charges(self, mmcharges):
        """set the values of point charges"""
        self.mmcharges = mmcharges
        self.updated = True

    def get_forces(self, calc):
        """forces acting on point charges"""
        self.mmforces = calc.get_forces_on_point_charges()
        return self.mmforces
