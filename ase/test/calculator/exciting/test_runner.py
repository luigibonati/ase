import pytest

from ase.calculators.exciting.runner import SimpleBinaryRunner, ExcitingRunner


@pytest.mark.parametrize(
    "binary_name, expected_run_cmd, expected_omp_num_threads, expected_directory, expected_timeout, expected_args",
    [("exciting_serial",  ['./'],                 1, './', 600, ['']),
     ("exciting_purempi", ['mpirun', '-np', '2'], 1, './', 600, ['']),
     ("exciting_smp",     ['./'],                 4, './', 600, ['']),
     ("exciting_mpismp",  ['mpirun', '-np', '2'], 2, './', 600, [''])
     ])
def test_class_exciting_runner_binary_defaults(tmpdir, binary_name,
                                               expected_run_cmd,
                                               expected_omp_num_threads,
                                               expected_directory,
                                               expected_timeout,
                                               expected_args):
    """
    Valid binary names and corresponding defaults
    """
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


def test_class_exciting_runner_no_defaults_with_binary_alias(tmpdir):
    """
    Binary alias does not have specified default run settings
    """
    binary = tmpdir / "exciting"
    binary.write("Arbitrary text such that file exists")

    with pytest.raises(KeyError) as error_info:
        runner = ExcitingRunner(binary)
    assert error_info.value.args[0] == "No default settings exist for this binary choice: exciting"
