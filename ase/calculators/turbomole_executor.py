from subprocess import Popen, PIPE


def execute(args, input_str=None, error_test=True,
            stdout_tofile=True):
    """executes a turbomole executable and process the outputs"""

    if isinstance(args, str):
        args = args.split()

    if stdout_tofile:
        stdout_file = 'ASE.TM.' + args[0] + '.out'
        stdout = open(stdout_file, 'w')
    else:
        stdout = PIPE

    if input_str:
        stdin = input_str.encode()
    else:
        stdin = None

    message = 'TM command "' + args[0] + '" execution failed'
    try:
        proc = Popen(args, stdin=PIPE, stderr=PIPE, stdout=stdout)
        res = proc.communicate(input=stdin)
        if error_test:
            error = res[1].decode()
            if 'abnormally' in error or 'ended normally' not in error:
                message += ' with error:\n' + error
                message += '\nSee file ' + stdout_file + ' for details.\n'
                raise RuntimeError(message)
    except RuntimeError as err:
        if stdout_tofile:
            stdout.close()
        raise err
    except OSError as err:
        if stdout_tofile:
            stdout.close()
        raise OSError(err.args[1] + '\n' + message)
    else:
        print('TM command: "' + args[0] + '" successfully executed')

    if stdout_tofile:
        stdout.close()
    else:
        return res[0].decode()
