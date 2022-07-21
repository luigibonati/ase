import json
import os
import re
import sys
from pprint import pprint
from subprocess import run


def exec_and_check_modules(expression):
    # Take null outside command to avoid
    # `import os` before expression
    null = os.devnull
    command = ("import sys;"
               f" stdout = sys.stdout; sys.stdout = open({repr(null)}, 'w');"
               f" {expression};"
               f" sys.stdout = stdout;"
               " modules = list(sys.modules.keys());"
               " import json; print(json.dumps(modules))")
    proc = run([sys.executable, '-c', command],
               capture_output=True, text=True, check=True)
    return set(json.loads(proc.stdout))


def check_imports(expression, *,
                  forbidden_modules=[],
                  max_module_count=10000,
                  do_print=False):
    modules = exec_and_check_modules(expression)

    if do_print:
        print('modules:')
        pprint(sorted(modules))

    for module_pattern in forbidden_modules:
        r = re.compile(module_pattern)
        for module in modules:
            assert not r.fullmatch(module), \
                f'{module} was imported'

    module_count = len(modules)
    assert module_count <= max_module_count, \
        f'too many modules loaded {module_count}/{max_module_count}'


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('expression')
    parser.add_argument('--forbidden_modules', nargs='+', default=[])
    parser.add_argument('--max_module_count', type=int, default=10000)
    parser.add_argument('--do_print', action='store_true')
    args = parser.parse_args()

    check_imports(**vars(args))
