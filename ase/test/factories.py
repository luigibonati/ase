import os
from pathlib import Path
from typing import Mapping
import configparser

import pytest

from ase.calculators.calculator import (names as calculator_names,
                                        get_calculator_class)


class NotInstalled(Exception):
    pass


def get_testing_executables():
    # TODO: better cross-platform support (namely Windows),
    # and a cross-platform global config file like /etc/ase/ase.conf
    paths = [Path.home() / '.config' / 'ase' / 'ase.conf']
    try:
        paths += [Path(x) for x in os.environ['ASE_CONFIG'].split(':')]
    except KeyError:
        pass
    conf = configparser.ConfigParser()
    conf['executables'] = {}
    effective_paths = conf.read(paths)
    return effective_paths, conf['executables']


factory_classes = {}


def factory(name):
    def decorator(cls):
        cls.name = name
        factory_classes[name] = cls
        return cls
    return decorator


def make_factory_fixture(name):
    @pytest.fixture(scope='session')
    def _factory(factories):
        if not factories.installed(name):
            pytest.skip(f'Not installed: {name}')
        return factories[name]
    _factory.__name__ = '{}_factory'.format(name)
    return _factory


@factory('abinit')
class AbinitFactory:
    def __init__(self, executable, pp_paths):
        self.executable = executable
        self.pp_paths = pp_paths

    def _base_kw(self):
        command = '{} < PREFIX.files > PREFIX.log'.format(self.executable)
        return dict(command=command,
                    pp_paths=self.pp_paths,
                    ecut=150,
                    chksymbreak=0,
                    toldfe=1e-3)

    def calc(self, **kwargs):
        from ase.calculators.abinit import Abinit
        kw = self._base_kw()
        kw.update(kwargs)
        return Abinit(**kw)

    @classmethod
    def fromconfig(cls, config):
        return AbinitFactory(config.executables['abinit'],
                             config.datafiles['abinit'])


@factory('asap')
class AsapFactory:
    importname = 'asap3'

    def calc(self, **kwargs):
        from asap3 import EMT
        return EMT(**kwargs)

    @classmethod
    def fromconfig(cls, config):
        # XXXX TODO Clean this up.  Copy of GPAW.
        # How do we design these things?
        import importlib
        spec = importlib.util.find_spec('asap3')
        if spec is None:
            raise NotInstalled('asap3')
        return cls()


@factory('cp2k')
class CP2KFactory:
    def __init__(self, executable):
        self.executable = executable

    def calc(self, **kwargs):
        from ase.calculators.cp2k import CP2K
        return CP2K(command=self.executable, **kwargs)

    @classmethod
    def fromconfig(cls, config):
        return CP2KFactory(config.executables['cp2k'])


@factory('dftb')
class DFTBFactory:
    def __init__(self, executable):
        self.executable = executable

    def calc(self, **kwargs):
        from ase.calculators.dftb import Dftb
        # XXX datafiles should be imported from datafiles project
        # We should include more datafiles for DFTB there, and remove them
        # from ASE's own datadir.
        command = f'{self.executable} > PREFIX.out'
        datadir = Path(__file__).parent / 'testdata'
        assert datadir.exists()
        return Dftb(command=command,
                    slako_dir=str(datadir) + '/',  # XXX not obvious
                    **kwargs)

    @classmethod
    def fromconfig(cls, config):
        return cls(config.executables['dftb'])


@factory('espresso')
class EspressoFactory:
    def __init__(self, executable, pseudo_dir):
        self.executable = executable
        self.pseudo_dir = pseudo_dir

    def _base_kw(self):
        from ase.units import Ry
        return dict(ecutwfc=300 / Ry)

    def calc(self, **kwargs):
        from ase.calculators.espresso import Espresso
        command = '{} -in PREFIX.pwi > PREFIX.pwo'.format(self.executable)
        pseudopotentials = {}
        for path in self.pseudo_dir.glob('*.UPF'):
            fname = path.name
            # Names are e.g. si_lda_v1.uspp.F.UPF
            symbol = fname.split('_', 1)[0].capitalize()
            pseudopotentials[symbol] = fname

        kw = self._base_kw()
        kw.update(kwargs)
        return Espresso(command=command, pseudo_dir=str(self.pseudo_dir),
                        pseudopotentials=pseudopotentials,
                        **kw)

    @classmethod
    def fromconfig(cls, config):
        paths = config.datafiles['espresso']
        assert len(paths) == 1
        return cls(config.executables['espresso'], paths[0])


@factory('gpaw')
class GPAWFactory:
    importname = 'gpaw'

    def calc(self, **kwargs):
        from gpaw import GPAW
        return GPAW(**kwargs)

    @classmethod
    def fromconfig(cls, config):
        import importlib
        spec = importlib.util.find_spec('gpaw')
        # XXX should be made non-pytest dependent
        if spec is None:
            raise NotInstalled('gpaw')
        return cls()


class BuiltinCalculatorFactory:
    def calc(self, **kwargs):
        from ase.calculators.calculator import get_calculator_class
        cls = get_calculator_class(self.name)
        return cls(**kwargs)

    @classmethod
    def fromconfig(cls, config):
        return cls()


@factory('emt')
class EMTFactory(BuiltinCalculatorFactory):
    pass


@factory('lammpsrun')
class LammpsRunFactory:
    def __init__(self, executable):
        self.executable = executable

    def calc(self, **kwargs):
        from ase.calculators.lammpsrun import LAMMPS
        return LAMMPS(command=self.executable, **kwargs)

    @classmethod
    def fromconfig(cls, config):
        return cls(config.executables['lammps'])


@factory('octopus')
class OctopusFactory:
    def __init__(self, executable):
        self.executable = executable

    def calc(self, **kwargs):
        from ase.calculators.octopus import Octopus
        command = f'{self.executable} > stdout.log'
        return Octopus(command=command, **kwargs)

    @classmethod
    def fromconfig(cls, config):
        return cls(config.executables['octopus'])


@factory('siesta')
class SiestaFactory:
    def __init__(self, executable, pseudo_path):
        self.executable = executable
        self.pseudo_path = pseudo_path

    def calc(self, **kwargs):
        from ase.calculators.siesta import Siesta
        command = '{} < PREFIX.fdf > PREFIX.out'.format(self.executable)
        return Siesta(command=command, pseudo_path=str(self.pseudo_path),
                      **kwargs)

    @classmethod
    def fromconfig(cls, config):
        paths = config.datafiles['siesta']
        assert len(paths) == 1
        path = paths[0]
        return cls(config.executables['siesta'], str(path))


@factory('nwchem')
class NWChemFactory:
    def __init__(self, executable):
        self.executable = executable

    def calc(self, **kwargs):
        from ase.calculators.nwchem import NWChem
        command = f'{self.executable} PREFIX.nwi > PREFIX.nwo'
        return NWChem(command=command, **kwargs)

    @classmethod
    def fromconfig(cls, config):
        return cls(config.executables['nwchem'])


class NoSuchCalculator(Exception):
    pass


class Factories:
    all_calculators = set(calculator_names)
    builtin_calculators = {'eam', 'emt', 'ff', 'lj', 'morse', 'tip3p', 'tip4p'}
    autoenabled_calculators = {'asap'} | builtin_calculators

    def __init__(self, requested_calculators):
        executable_config_paths, executables = get_testing_executables()
        assert isinstance(executables, Mapping), executables
        self.executables = executables
        self.executable_config_paths = executable_config_paths

        datafiles_module = None
        datafiles = {}

        try:
            import asetest as datafiles_module
        except ImportError:
            pass
        else:
            datafiles.update(datafiles_module.datafiles.paths)
            datafiles_module = datafiles_module

        self.datafiles_module = datafiles_module
        self.datafiles = datafiles

        factories = {}

        for name, cls in factory_classes.items():
            try:
                factory = cls.fromconfig(self)
            except (NotInstalled, KeyError):
                pass
            else:
                factories[name] = factory

        self.factories = factories

        requested_calculators = set(requested_calculators)
        if 'auto' in requested_calculators:
            requested_calculators.remove('auto')
            requested_calculators |= set(self.factories)
        self.requested_calculators = requested_calculators

        for name in self.requested_calculators:
            if name not in self.all_calculators:
                raise NoSuchCalculator(name)

    def installed(self, name):
        return name in self.builtin_calculators | set(self.factories)

    def is_adhoc(self, name):
        return name not in factory_classes

    def optional(self, name):
        return name not in self.builtin_calculators

    def enabled(self, name):
        auto = name in self.autoenabled_calculators and self.installed(name)
        return auto or (name in self.requested_calculators)

    def require(self, name):
        # XXX This is for old-style calculator tests.
        # Newer calculator tests would depend on a fixture which would
        # make them skip.
        # Older tests call require(name) explicitly.
        assert name in calculator_names
        if name not in self.requested_calculators:
            pytest.skip(f'use --calculators={name} to enable')

    def __getitem__(self, name):
        return self.factories[name]

    def monkeypatch_disabled_calculators(self):
        test_calculator_names = (self.autoenabled_calculators |
                                 self.builtin_calculators |
                                 self.requested_calculators)
        disable_names = self.all_calculators - test_calculator_names

        for name in disable_names:
            try:
                cls = get_calculator_class(name)
            except ImportError:
                pass
            else:
                def get_mock_init(name):
                    def mock_init(obj, *args, **kwargs):
                        pytest.skip(f'use --calculators={name} to enable')
                    return mock_init

                def mock_del(obj):
                    pass
                cls.__init__ = get_mock_init(name)
                cls.__del__ = mock_del


def get_factories(pytestconfig):
    opt = pytestconfig.getoption('--calculators')
    requested_calculators = opt.split(',') if opt else []
    return Factories(requested_calculators)


def parametrize_calculator_tests(metafunc):
    """Parametrize tests using our custom markers.

    We want tests marked with @pytest.mark.calculator(names) to be
    parametrized over the named calculator or calculators."""
    calculator_inputs = []

    for marker in metafunc.definition.iter_markers(name='calculator'):
        calculator_names = marker.args
        kwargs = dict(marker.kwargs)
        marks = kwargs.pop('marks', [])
        for name in calculator_names:
            param = pytest.param((name, kwargs), marks=marks)
            calculator_inputs.append(param)

    if calculator_inputs:
        metafunc.parametrize('factory', calculator_inputs, indirect=True,
                             ids=lambda input: input[0])


class CalculatorInputs:
    def __init__(self, factory, parameters=None):
        if parameters is None:
            parameters = {}
        self.parameters = parameters
        self.factory = factory

    @property
    def name(self):
        return self.factory.name

    def __repr__(self):
        cls = type(self)
        return '{}({}, {})'.format(cls.__name__,
                                   self.name, self.parameters)

    def new(self, **kwargs):
        kw = dict(self.parameters)
        kw.update(kwargs)
        return CalculatorInputs(self.factory, kw)

    def calc(self, **kwargs):
        param = dict(self.parameters)
        param.update(kwargs)
        return self.factory.calc(**param)


class ObsoleteFactoryWrapper:
    # We use this for transitioning older tests to the new framework.
    def __init__(self, name):
        self.name = name

    def calc(self, **kwargs):
        from ase.calculators.calculator import get_calculator_class
        cls = get_calculator_class(self.name)
        return cls(**kwargs)
