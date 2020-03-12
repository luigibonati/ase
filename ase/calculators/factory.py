import os
from typing import Any, Tuple
from functools import partial
import subprocess
import shlex
from configparser import NoOptionError

from ase.calculators.calculator import FileIOCalculator
from ase.calculators.gamess_us import GAMESSUS
from ase.calculators.espresso import Espresso
from ase.calculators.abinit import Abinit
from ase.calculators.nwchem import NWChem
from ase.config import config


_name_to_calc = {
    'gamess-us': GAMESSUS,
    'espresso': Espresso,
    'abinit': Abinit,
    'nwchem': NWChem,
}


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
    calculator = calculator.lower()
    Calc = _name_to_calc.get(calculator)
    if Calc is None:
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

    for option, keyword in Calc.config_parameters.items():
        if config.has_option(calculator, option):
            kwargs[keyword] = config.get(calculator, option)

    return Calc(launcher=partial(launcher, calculator, nproc), **kwargs)
