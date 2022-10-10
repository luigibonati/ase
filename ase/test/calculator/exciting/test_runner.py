"""Test runner classes to run exciting simulations using subproces."""
import logging
import pytest


LOGGER = logging.getLogger(__name__)


try:
    __import__('excitingtools')
    from ase.calculators.exciting.runner import (
        SimpleBinaryRunner, ExcitingRunner)

except ModuleNotFoundError:
    MESSAGE = "exciting tests are skipped if excitingtools not installed."
    LOGGER.info(MESSAGE)


@pytest.fixture
def excitingtools():
    """If we cannot import excitingtools we skip tests with this fixture."""
    return pytest.importorskip('excitingtools')


def test_class_simple_binary_runner(tmpdir, excitingtools):
    """Test SimpleBinaryRunner."""
    binary = tmpdir / 'binary.exe'
    binary.write("Arbitrary text such that file exists")
    runner = SimpleBinaryRunner(
        binary=binary, run_cmd=['mpirun', '-np', '2'], omp_num_threads=1,
        time_out=600, directory=tmpdir,
        args=['input.txt'])

    # Attributes
    assert runner.binary == binary
    assert runner.run_cmd == ['mpirun', '-np', '2']
    assert runner.omp_num_threads == 1
    assert runner.directory == tmpdir
    assert runner.time_out == 600
    assert runner.args == ['input.txt']

    # Methods
    execute = runner.compose_execution_list()
    assert execute == ['mpirun', '-np', '2', str(binary), 'input.txt']


@pytest.mark.parametrize(
    (
        "binary_name, expected_run_cmd, expected_omp_num_threads,"
        "expected_directory, expected_timeout, expected_args"), [
            ("exciting_serial", ['./'], 1, './', 600, ['']),
            ("exciting_purempi", ['mpirun', '-np', '2'], 1, './', 600, ['']),
            ("exciting_smp", ['./'], 4, './', 600, ['']),
            ("exciting_mpismp", ['mpirun', '-np', '2'], 2, './', 600, [''])
     ])
def test_class_exciting_runner_binary_defaults(
        tmpdir,
        binary_name,
        expected_run_cmd,
        expected_omp_num_threads,
        expected_directory,
        expected_timeout,
        expected_args,
        excitingtools):
    """Valid binary names and corresponding default attributes."""
    binary = tmpdir / binary_name
    binary.write("Arbitrary text such that file exists")
    runner = ExcitingRunner(binary)

    # Class attributes
    assert runner.binary == binary, "Binary erroneously initialised"
    assert runner.run_cmd == expected_run_cmd
    assert runner.omp_num_threads == expected_omp_num_threads
    assert runner.directory == expected_directory
    assert runner.time_out == expected_timeout
    assert runner.args == expected_args


def test_class_exciting_runner_no_defaults_with_binary_alias(
        tmpdir, excitingtools):
    """Binary alias does not have specified default run settings."""
    binary = tmpdir / "exciting"
    binary.write("Arbitrary text such that file exists")

    with pytest.raises(KeyError) as error_info:
        ExcitingRunner(binary)
    assert error_info.value.args[0] == (
        "No default settings exist for this binary choice: exciting")


def test_class_exciting_runner_erroneous_binary_name(tmpdir, excitingtools):
    """Binary name is not listed class `binaries` attribute."""
    binary = tmpdir / "exciting_erroneous_name"
    binary.write("Arbitrary text such that file exists")

    with pytest.raises(ValueError) as error_info:
        ExcitingRunner(binary)
    assert error_info.value.args[0] == (
        "binary name is not a valid choice: exciting_erroneous_name")
