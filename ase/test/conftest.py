from pathlib import Path
from subprocess import Popen, PIPE, check_output
import zlib

import pytest
import numpy as np

import ase
from ase.utils import workdir
from ase.test.factories import (Factories, CalculatorInputs, NotInstalled,
                                factory_classes, BuiltinCalculatorFactory,
                                make_factory_fixture, get_testing_executables)
from ase.calculators.calculator import (names as calculator_names,
                                        get_calculator_class)
from ase.dependencies import all_dependencies


@pytest.fixture(scope='session')
def enabled_calculators(pytestconfig):
    return get_enabled_calculators(pytestconfig)


def get_enabled_calculators(pytestconfig):
    opt = pytestconfig.getoption('--calculators')
    all_names = set(calculator_names)

    names = set(always_enabled_calculators)
    if opt:
        for name in opt.split(','):
            if name not in all_names:
                raise ValueError(f'No such calculator: {name}')
            names.add(name)
    return sorted(names)


def get_factories(pytestconfig):
    try:
        import asetest
    except ImportError:
        datafiles = {}
    else:
        datafiles = asetest.datafiles.paths

    testing_executables = get_testing_executables()
    enabled_calculators = get_enabled_calculators(pytestconfig)
    return Factories(testing_executables, datafiles, enabled_calculators)


def pytest_report_header(config, startdir):
    messages = []

    def add(msg=''):
        messages.append(msg)

    add()
    add('Libraries')
    add('=========')
    add()
    for name, path in all_dependencies():
        add('{:24} {}'.format(name, path))
    add()

    add('Calculators')
    add('===========')
    add()
    calculators_option = config.getoption('--calculators')
    if calculators_option:
        requested_calculators = set(calculators_option.split(','))
    else:
        requested_calculators = set()

    for name in requested_calculators:
        if name not in calculator_names:
            pytest.exit(f'No such calculator: {name}')

    factories = get_factories(config)
    available_calculators = set()

    for name in sorted(factory_classes):
        cls = factory_classes[name]
        if issubclass(cls, BuiltinCalculatorFactory):
            # Not interesting to test presence of always-present calculators.
            continue

        try:
            factory = cls.fromconfig(factories)
        except (KeyError, NotInstalled):
            factory = None
        else:
            available_calculators.add(name)

        if factory is None:
            configinfo = 'not installed'
        else:
            # Some really ugly hacks here:
            if hasattr(factory, 'importname'):
                import importlib
                module = importlib.import_module(factory.importname)
                configinfo = str(module.__path__[0])  # type: ignore
            else:
                configtokens = []
                for varname, variable in vars(factory).items():
                    configtokens.append(f'{varname}={variable}')
                configinfo = ', '.join(configtokens)

        run = '[x]' if name in requested_calculators else '[ ]'
        line = f'  {run} {name:10} {configinfo}'
        add(line)
    add()

    for name in requested_calculators:
        if name in factory_classes and name not in available_calculators:
            pytest.exit(f'Calculator {name} is not installed.  '
                        'Please run "ase test --help-calculators".')

    return messages


@pytest.fixture(scope='session')
def require_vasp(factories):
    factories.require('vasp')


def disable_calculators(names):
    for name in names:
        if name in always_enabled_calculators:
            continue
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


# asap is special, being the only calculator that may not be installed.
# But we want that for performance in some tests.
always_enabled_calculators = set(
    ['asap', 'eam', 'emt', 'ff', 'lj', 'morse', 'tip3p', 'tip4p']
)


@pytest.fixture(scope='session', autouse=True)
def monkeypatch_disabled_calculators(request, enabled_calculators):
    test_calculator_names = list(always_enabled_calculators)
    test_calculator_names += enabled_calculators
    disable_calculators([name for name in calculator_names
                         if name not in enabled_calculators])


@pytest.fixture(autouse=True)
def use_tmp_workdir(tmp_path):
    # Pytest can on some systems provide a Path from pathlib2.  Normalize:
    path = Path(str(tmp_path))
    with workdir(path, mkdir=True):
        yield tmp_path
    print(f'Testpath: {path}')


@pytest.fixture(scope='session')
def tkinter():
    import tkinter
    try:
        tkinter.Tk()
    except tkinter.TclError as err:
        pytest.skip('no tkinter: {}'.format(err))


@pytest.fixture(scope='session')
def plt(tkinter):
    # XXX Probably we can get rid of tkinter requirement.
    matplotlib = pytest.importorskip('matplotlib')
    matplotlib.use('Agg')

    import matplotlib.pyplot as plt
    return plt


@pytest.fixture
def figure(plt):
    fig = plt.figure()
    yield fig
    plt.close(fig)


@pytest.fixture(scope='session')
def psycopg2():
    return pytest.importorskip('psycopg2')


@pytest.fixture(scope='session')
def factories(pytestconfig):
    return get_factories(pytestconfig)


abinit_factory = make_factory_fixture('abinit')
cp2k_factory = make_factory_fixture('cp2k')
dftb_factory = make_factory_fixture('dftb')
espresso_factory = make_factory_fixture('espresso')
gpaw_factory = make_factory_fixture('gpaw')
octopus_factory = make_factory_fixture('octopus')
siesta_factory = make_factory_fixture('siesta')


@pytest.fixture
def factory(request, factories):
    name, kwargs = request.param
    try:
        factory = factories[name]
    except NotInstalled:
        pytest.skip(f'Not installed: {name}')
    return CalculatorInputs(factory, kwargs)


def pytest_generate_tests(metafunc):
    from ase.test.factories import parametrize_calculator_tests
    parametrize_calculator_tests(metafunc)

    if 'seed' in metafunc.fixturenames:
        seeds = metafunc.config.getoption('seed')
        if len(seeds) == 0:
            seeds = [0, 1]
        else:
            seeds = list(map(int, seeds))
        metafunc.parametrize('seed', seeds)


class CLI:
    def __init__(self, calculators):
        self.calculators = calculators

    def ase(self, *args):
        proc = Popen(['ase', '-T'] + list(args),
                     stdout=PIPE, stdin=PIPE)
        stdout, _ = proc.communicate(b'')
        status = proc.wait()
        assert status == 0
        return stdout.decode('utf-8')

    def shell(self, command, calculator_name=None):
        if calculator_name is not None:
            self.calculators.require(calculator_name)

        actual_command = ' '.join(command.split('\n')).strip()
        output = check_output(actual_command, shell=True)
        return output.decode()


@pytest.fixture(scope='session')
def cli(factories):
    return CLI(factories)


@pytest.fixture(scope='session')
def datadir():
    test_basedir = Path(__file__).parent
    return test_basedir / 'testdata'


@pytest.fixture
def pt_eam_potential_file(datadir):
    # EAM potential for Pt from LAMMPS, also used with eam calculator.
    # (Where should this fixture really live?)
    return datadir / 'eam_Pt_u3.dat'


@pytest.fixture(scope='session')
def asap3():
    return pytest.importorskip('asap3')


@pytest.fixture(autouse=True)
def arbitrarily_seed_rng(request):
    # We want tests to not use global stuff such as np.random.seed().
    # But they do.
    #
    # So in lieu of (yet) fixing it, we reseed and unseed the random
    # state for every test.  That makes each test deterministic if it
    # uses random numbers without seeding, but also repairs the damage
    # done to global state if it did seed.
    #
    # In order not to generate all the same random numbers in every test,
    # we seed according to a kind of hash:
    ase_path = ase.__path__[0]
    abspath = Path(request.module.__file__)
    relpath = abspath.relative_to(ase_path)
    module_identifier = str(relpath)
    function_name = request.function.__name__
    hashable_string = f'{module_identifier}:{function_name}'
    # We use zlib.adler32() rather than hash() because Python randomizes
    # the string hashing at startup for security reasons.
    seed = zlib.adler32(hashable_string.encode('ascii')) % 12345
    # (We should really use the full qualified name of the test method.)
    state = np.random.get_state()
    np.random.seed(seed)
    yield
    np.random.set_state(state)


def pytest_addoption(parser):
    parser.addoption('--calculators', metavar='NAMES', default='',
                     help='comma-separated list of calculators to test')
    parser.addoption('--seed', action='append', default=[],
                     help='add a seed for tests where random number generators'
                          ' are involved. This option can be applied more'
                          ' than once.')
