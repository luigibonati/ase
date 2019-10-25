import os
from unittest import SkipTest
from ase.calculators.launcher import Launcher

if os.name != 'posix':
    raise SkipTest('Only on posix os')

cmd = Launcher('echo', ['hello', 'world'], stdout='/dev/null')
print('command', cmd.as_shell_command())
status = cmd.popen().wait()
assert status == 0

cmd.stdout = 'out.txt'
cmd.stdin = '/dev/null'
print(cmd.as_shell_command())
status = cmd.popen().wait()
assert status == 0
with open('out.txt') as fd:
    assert fd.read().strip() == 'hello world'

cmd.cores = 4
cmd.parallel_args = ['mpiexec', '-np', '{cores}']
cmd.stderr = 'err.txt'

# We may not have mpi, but we can check that we evaluate to the right command:
ref = 'mpiexec -np 4 echo hello world < /dev/null 1> out.txt 2> err.txt'
assert cmd.as_shell_command().split() == ref.split()
