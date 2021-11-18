"""
Binary runner and results class
"""
from typing import List, Optional
from pathlib import Path
import sys
import os
import subprocess


class SubprocessRunResults:
    """
    Results returned from subprocess.run
    """

    def __init__(self, stdout, stderr, return_code: int):
        self.stdout = stdout
        self.stderr = stderr
        self.return_code = return_code
        self.success = return_code == 0


class SimpleBinaryRunner:
    """
    Compose a run command and run a binary
    """

    def __init__(self,
                 binary: str,
                 run_cmd: List[str],
                 omp_num_threads: int,
                 time_out: int,
                 directory='./',
                 args: Optional[List[str]] = ['']
                 ) -> None:
        """
        :param str binary: Binary name
        :param List[str] run_cmd: Run commands sequentially as a list. For example:
          * For serial: ['./']
          * For MPI:   ['mpirun', '-np', '2']
        :param int omp_num_threads: Number of OMP threads
        :param int time_out: Number of seconds before a job is defined to have timed out
        :param List[str] args: Optional binary arguments
        """
        self.binary = binary
        self.directory = directory
        self.run_cmd = run_cmd
        self.omp_num_threads = omp_num_threads
        self.args = args
        self.time_out = time_out

        assert omp_num_threads > 0, "Number of OMP threads must be > 0"

        try:
            i = run_cmd.index('np')
            mpi_processes = eval(run_cmd[i + 1])
            assert type(mpi_processes) == int, "Number of MPI processes should be an int"
            assert mpi_processes > 0, "Number of MPI processes must be > 0"
        except ValueError:
            pass

    def compose_execution_list(self) -> list:
        """
        Generate a complete list of strings to pass to subprocess.run, to execute the calculation.

        For example, given:
          ['mpirun', '-np, '2'] + ['binary.exe'] + ['>', 'std.out']

        return ['mpirun', '-np, '2', 'binary.exe', '>', 'std.out']
        """
        return self.run_cmd + [self.binary] + self.args

    def run(self, directory: Optional[str] = None, execution_list: Optional[list] = None) -> SubprocessRunResults:
        """
        Run a binary.

        :param str directory: Optional Directory in which to run the execute command.
        :param Optional[list] execution_list: Optional List of arguments required by subprocess.run. Defaults to None.
        """

        if directory is not None:
            directory = self.directory

        # TODO(Alex) What is the best thing to do in this case?
        if not Path(directory).is_dir():
            sys.exit('Directory does not exist:', directory)

        if execution_list is None:
            execution_list = self.compose_execution_list()

        my_env = {**os.environ, "OMP_NUM_THREADS": str(self.omp_num_threads)}
        result = subprocess.run(execution_list,
                                env=my_env,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                timeout=self.time_out,
                                cwd=directory)

        return SubprocessRunResults(result.stdout, result.stderr, result.returncode)


class ExcitingRunner(SimpleBinaryRunner):
    """
    Execute an exciting calculation using a simple binary runner.
    """

    # Exciting has one input file
    input_name = 'input.xml'

    binary_exts = ['serial', 'purempi', 'smp', 'mpismp']
    binaries = ['exciting_' + ext for ext in binary_exts]

    default_run_cmd = {'exciting_serial': ['./'],
                       'exciting_purempi': ['mpirun', '-np', '2'],
                       'exciting_smp': ['./'] ,
                       'exciting_mpismp': ['mpirun', '-np', '2']
                       }

    default_omp_threads = {'exciting_serial': 1,
                           'exciting_purempi': 1,
                           'exciting_smp': 4,
                           'exciting_mpismp': 2
                           }

    def __init__(self,
                 binary: str,
                 run_cmd: Optional[List[str]] = None,
                 omp_num_threads: Optional[int] = None,
                 time_out: Optional[int] = 600,
                 directory: Optional[str] = './',
                 args: Optional[List[str]] = [''],
                 ) -> None:

        if run_cmd is None:
            run_cmd = self.default_run_cmd[binary]

        if omp_num_threads is None:
            omp_num_threads = self.default_omp_threads[binary]

        assert binary in self.binaries, "binary string is not a valid choice"
        super().__init__(binary, run_cmd, omp_num_threads, time_out, directory, args)
