"""
Execution of turbomole binaries and scripts:
define, dscf, dgrad, ridft, rdgrad, aoforce, eiger/eigerf, sdg, kdg, jobex,
NumForce
"""
from subprocess import Popen, PIPE


def check_bad_output(stderr):
    if 'abnormally' in stderr or 'ended normally' not in stderr:
        raise OSError(f'Turbomole error: {stderr}')


def execute(args, input_str=''):
    """executes a turbomole executable and process the outputs"""
    if isinstance(args, str):
        args = args.split()
    # XXX Can we remove this?  And only have lists of strings as input.

    stdout_file = 'ASE.TM.' + args[0] + '.out'
    with open(stdout_file, 'w') as stdout:
        proc = Popen(args, stdin=PIPE, stderr=PIPE, stdout=stdout,
                     encoding='ASCII')
        stdout_txt, stderr_txt = proc.communicate(input=input_str)
        check_bad_output(stderr_txt)
    return stdout_file


def test_execute():
    # XXX Move to test suite
    python = sys.executable
    msg = 'hello world'
    stdout_file = execute([python, '-c', 'print("{}")'])
    with open(stdout_file) as fd:
        assert fd.read() == msg
