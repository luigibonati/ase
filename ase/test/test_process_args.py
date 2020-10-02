import pytest
from ase.utils.process_args import parse_command, ProcessArgs


@pytest.mark.parametrize('cmd, ref_normalized', [
    ('arg', 'arg'),
    ('arg>out', 'arg > out'),
    ('arg1>out', 'arg1 > out'),
    ('arg2>out', 'arg2 > out'),
    ('arg 1>out', 'arg > out'),
    ('arg 2>out', 'arg 2> out'),
    ('arg<in', 'arg < in'),
    ('arg<in 2>out2 1> out1', 'arg < in > out1 2> out2'),
])
def test_parse_command(cmd, ref_normalized):
    cmd_obj = parse_command(cmd)
    print(cmd, '-->', cmd_obj)
    normalized_command = cmd_obj.as_shell()
    assert normalized_command == ref_normalized


@pytest.mark.parametrize('cmd, errmsg', [
    ('arg "hello world"', 'avoid characters'),
    ('arg 1> aa > bb', 'redirected twice'),
    ('arg >', 'Missing argument'),
])
def test_parse_bad_command(cmd, errmsg):
    with pytest.raises(ValueError, match=errmsg):
        print(parse_command(cmd))


def test_process_args():
    args = ProcessArgs(['a', 'b'], stdin='x', stdout='y', stderr='z')
    shellcommand = 'a b < x > y 2> z'
    assert args.as_shell() == shellcommand
    assert shellcommand in str(args)

