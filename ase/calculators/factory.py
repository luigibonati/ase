import os
from typing import Any, Tuple
from functools import partial
import subprocess
import shlex
from configparser import NoOptionError

from ase.calculators.calculator import FileIOCalculator
from ase.calculators.gamess_us import GAMESSUS
from ase.calculators.espresso import Espresso
from ase.config import config


def launcher(name: str, nproc: int, cwd: str = '.',
             prefix: str = None) -> Tuple[int, str]:
    if prefix is None:
        prefix = name

    command = config.get(name, 'command').format(prefix=prefix, nproc=nproc)
    try:
        stdin = config.get(name, 'stdin').format(prefix=prefix)
    except NoOptionError:
        stdin = os.devnull

    stdout = config.get(name, 'stdout').format(prefix=prefix)

    with open(stdin, 'r') as fin:
        with open(stdout, 'w') as fout:
            proc = subprocess.run(shlex.split(command), stdin=fin, stdout=fout,
                                  shell=False, cwd=cwd)

    return proc.returncode, command


def get_calculator(
        calculator: str,
        nproc: int = 1,
        profile: str = None,
        **kwargs: Any
) -> FileIOCalculator:

    if calculator == 'gamess-us':
        Calc = GAMESSUS
    elif calculator == 'espresso':
        Calc = Espresso
    else:
        raise ValueError('Unknown calculator {}'.format(calculator))

    if not config.has_section(calculator):
        raise RuntimeError('No configuration options found for calculator {}'
                           .format(calculator))

    if profile is not None:
        name = '_'.join([calculator, profile])
        if not config.has_section(name):
            raise RuntimeError('No profile named {} was found for calculator '
                               '{}'.format(profile, calculator))
        calculator = name

    return Calc(launcher=partial(launcher, calculator, nproc), **kwargs)
