import pytest
import re


redirection_symbols = {'<': 'stdin',
                       '>': 'stdout',
                       '1>': 'stdout',
                       '2>': 'stderr'}
bad_chars = '"\'|()&'  # We are cowards and won't parse complex commands.


def split_redirection_chars_and_spaces(command):
    for token in re.split(r'(\s+1\>|\s+2\>|\s+|\<|\>)', command):
        token = token.strip()
        if token:
            yield token


class Command:
    def __init__(self, argv, stdin, stdout, stderr):
        self.argv = argv
        self.stdin = stdin
        self.stdout = stdout
        self.stderr = stderr

    def __repr__(self):
        return 'Command({})'.format(self.as_shell())

    def as_shell(self):
        argv = list(self.argv)
        if self.stdin:
            argv += ['<', self.stdin]
        if self.stdout:
            argv += ['>', self.stdout]
        if self.stderr:
            argv += ['2>', self.stderr]
        return ' '.join(argv)


def parse_command(command):
    if any(char in command for char in bad_chars):
        raise ValueError('Please avoid characters {bad_chars} in command')

    argv = []
    redirections = {'stdin': None, 'stdout': None, 'stderr': None}

    stdin = None
    stdout = None
    stderr = None
    itokens = split_redirection_chars_and_spaces(command)
    #itokens = iter(tokenize(command))
    for token in itokens:
        if token in redirection_symbols:
            if token in redirections:
                raise ValueError('Same stream redirected twice')
            stream_name = redirection_symbols[token]
            try:
                redirections[stream_name] = next(itokens)
            except StopIteration:
                raise ValueError('Missing argument for redirection')
        else:
            argv.append(token)
    return Command(argv, **redirections)


@pytest.mark.parametrize('cmd, ref', [
    ('arg>out', 'arg > out'),
    ('arg1>out', 'arg1 > out'),
    ('arg2>out', 'arg2 > out'),
    ('arg 1>out', 'arg > out'),
    ('arg 2>out', 'arg 2> out'),
    ('arg<in', 'arg < in'),
    ('arg<in 2>out2 1> out1', 'arg < in > out1 2> out2'),
])
def test_parse_command(cmd, ref):
    cmd_obj = parse_command(cmd)
    print(cmd, '-->', cmd_obj)
    normalized_command = cmd_obj.as_shell()
    assert normalized_command == ref
