# Run flake8 on main source dir and documentation.
import sys
from collections import defaultdict
from pathlib import Path
from subprocess import Popen, PIPE


max_errors = {
    # do not compare types, use 'isinstance()'
    'E721': 0,
    # multiple imports on one line
    'E401': 0,
    # multiple spaces before keyword
    'E272': 0,
    # continuation line under-indented for hanging indent
    'E121': 0,
    # whitespace before '('
    'E211': 0,
    # continuation line with same indent as next logical line
    'E125': 0,
    # comparison to True should be 'if cond is True:' or 'if cond:'
    'E712': 0,
    # 'name' imported but unused
    'F401': 0,
    # no newline at end of file
    'W292': 0,
    # missing whitespace after keyword
    'E275': 0,
    # multiple spaces after operator
    'E222': 0,
    # missing whitespace around modulo operator
    'E228': 0,
    # expected 1 blank line before a nested definition, found 0
    'E306': 0,
    # test for membership should be 'not in'
    'E713': 0,
    # multiple statements on one line (colon)
    'E701': 0,
    # indentation is not a multiple of four (comment)
    'E114': 0,
    # unexpected indentation (comment)
    'E116': 0,
    # comparison to None should be 'if cond is None:'
    'E711': 0,
    # expected 1 blank line, found 0
    'E301': 0,
    # multiple spaces after keyword
    'E271': 0,
    # test for object identity should be 'is not'
    'E714': 0,
    # closing bracket does not match visual indentation
    'E124': 0,
    # too many leading '#' for block comment
    'E266': 0,
    # over-indented
    'E117': 0,
    # indentation contains mixed spaces and tabs
    'E101': 0,
    # indentation contains tabs
    'W191': 0,
    # closing bracket does not match indentation of opening bracket's line
    'E123': 0,
    # multiple spaces before operator
    'E221': 0,
    # whitespace before '}'
    'E202': 0,
    # whitespace after '{'
    'E201': 0,
    # inline comment should start with '# '
    'E262': 12,
    # the backslash is redundant between brackets
    'E502': 0,
    # continuation line missing indentation or outdented
    'E122': 0,
    # indentation is not a multiple of four
    'E111': 28,
    # do not use bare 'except'
    'E722': 0,
    # whitespace before ':'
    'E203': 30,
    # blank line at end of file
    'W391': 30,
    # continuation line over-indented for hanging indent
    'E126': 27,
    # multiple spaces after ','
    'E241': 35,
    # continuation line under-indented for visual indent
    'E128': 39,
    # continuation line over-indented for visual indent
    'E127': 32,
    # missing whitespace around operator
    'E225': 43,
    # ambiguous variable name 'O'
    'E741': 77,
    # too many blank lines (2)
    'E303': 188,
    # expected 2 blank lines after class or function definition, found 1
    'E305': 35,
    # module level import not at top of file
    'E402': 0,
    # at least two spaces before inline comment
    'E261': 71,
    # expected 2 blank lines, found 1
    'E302': 102,
    # unexpected spaces around keyword / parameter equals
    'E251': 70,
    # trailing whitespace
    'W291': 169,
    # block comment should start with '# '
    'E265': 172,
    # missing whitespace after ','
    'E231': 300,
    # missing whitespace around arithmetic operator
    'E226': 308,
    # line too long (93 > 79 characters)
    'E501': 740}


def run_flaketest():
    asepath = Path(__file__).parent.parent
    if not (asepath.parent / 'setup.py').exists():
        raise RuntimeError(f'{asepath} not root in source tree?')

    args = [
        sys.executable,
        '-m',
        'flake8',
        str(asepath),
        str((asepath / '../doc').resolve()),
        '--exclude',
        str((asepath / '../doc/build/*').resolve()),
        '--ignore',
        'E129,W293,W503,W504,E741',
        '-j',
        '1'
    ]
    proc = Popen(args, stdout=PIPE)
    stdout, stderr = proc.communicate()
    stdout = stdout.decode('utf8')

    errors = defaultdict(int)
    files = defaultdict(int)
    offenses = defaultdict(list)
    descriptions = {}
    for stdout_line in stdout.splitlines():
        tokens = stdout_line.split(':', 3)
        filename, _, colno, complaint = tokens
        e = complaint.strip().split()[0]
        errors[e] += 1
        descriptions[e] = complaint
        files[filename] += 1
        offenses[e] += [stdout_line]

    errmsg = ''
    for err, nerrs in errors.items():
        nmaxerrs = max_errors.get(err, 0)
        if nerrs <= nmaxerrs:
            continue
        errmsg += 'Too many flakes: {} (max={})\n'.format(nerrs, nmaxerrs)
        errmsg += 'Offenses:\n' + '\n'.join(offenses[err]) + '\n'
    return errmsg


def main():
    # Run the flaketests directly on the source, without ASE installed.
    # We like that for the lint pipeline job.
    errmsg = run_flaketest()

    if errmsg:
        print('Flake test failed')
        print(errmsg)
        raise SystemExit(1)

    print('Flake test passed')


if __name__ == '__main__':
    main()
