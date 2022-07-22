import json
import os
import re
import sys
from pprint import pprint
from subprocess import run, PIPE


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
               # For Python 3.6 and possibly older
               stdout=PIPE, stderr=PIPE, universal_newlines=True,
               # For Python 3.7+ the next line is equivalent
               # capture_output=True, text=True,
               check=True)
    return set(json.loads(proc.stdout))


def check_imports(expression, *,
                  forbidden_modules=[],
                  max_module_count=None,
                  max_nonstdlib_module_count=None,
                  do_print=False):
    modules = exec_and_check_modules(expression)

    if do_print:
        print('all modules:')
        pprint(sorted(modules))

    for module_pattern in forbidden_modules:
        r = re.compile(module_pattern)
        for module in modules:
            assert not r.fullmatch(module), \
                f'{module} was imported'

    if max_nonstdlib_module_count is not None:
        assert sys.version_info >= (3, 10), 'Python 3.10+ required'

        nonstdlib_modules = []
        for module in modules:
            if module.split('.')[0] in sys.stdlib_module_names:
                continue
            nonstdlib_modules.append(module)

        if do_print:
            print('nonstdlib modules:')
            pprint(sorted(nonstdlib_modules))

        module_count = len(nonstdlib_modules)
        assert module_count <= max_nonstdlib_module_count, (
            'too many nonstdlib modules loaded:'
            f' {module_count}/{max_nonstdlib_module_count}'
        )

    if max_module_count is not None:
        module_count = len(modules)
        assert module_count <= max_module_count, \
            f'too many modules loaded: {module_count}/{max_module_count}'


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('expression')
    parser.add_argument('--forbidden_modules', nargs='+', default=[])
    parser.add_argument('--max_module_count', type=int, default=None)
    parser.add_argument('--max_nonstdlib_module_count', type=int, default=None)
    parser.add_argument('--do_print', action='store_true')
    args = parser.parse_args()

    check_imports(**vars(args))
