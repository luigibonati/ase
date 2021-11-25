import pytest

from ase.calculators.exciting.runner import SimpleBinaryRunner, ExcitingRunner


def test_class_exciting_runner_binary_options(tmpdir):
    """
    Valid binary names
    """
    binary = tmpdir / "exciting_serial"
    binary.write("Arbitrary text such that file exists")
    runner = ExcitingRunner(binary)
    assert runner.binary == binary, "Binary erroneously initialised"
    assert runner.run_cmd == ['./']
    assert runner.omp_num_threads == 1
    assert runner.directory == './'
    assert runner.time_out == 600
    assert runner.args == ['']


def test_class_exciting_runner_no_default_with_binary_alias(tmpdir):
    """
    Binary alias does not have specified default run settings
    """
    binary = tmpdir / "exciting"
    binary.write("Arbitrary text such that file exists")

    with pytest.raises(KeyError) as error_info:
        runner = ExcitingRunner(binary)
    assert error_info.value.args[0] == "No default settings exist for this binary choice: exciting"

