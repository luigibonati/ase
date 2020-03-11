import os
from typing import Dict, Any, Tuple
import subprocess
import shlex
from configparser import NoOptionError

from ase.calculators.calculator import FileIOCalculator
from ase.calculators.gamess_us import GAMESSUS
from ase.calculators.espresso import Espresso
from ase.config import config


class Launcher:
    def __init__(self, name: str, nproc: int) -> None:
        self.name = name
        self.nproc = nproc

    def run(self,
            cwd: str = '.',
            prefix: str = None) -> Tuple[int, str]:
        if prefix is None:
            prefix = self.name

        command = config.get(self.name, 'command').format(
            prefix=prefix, nproc=self.nproc,
        )

        try:
            stdin = config.get(self.name, 'stdin').format(prefix=prefix)
        except NoOptionError:
            stdin = os.devnull

        stdout = config.get(self.name, 'stdout').format(prefix=prefix)

        with open(stdin, 'r') as fin:
            with open(stdout, 'w') as fout:
                proc = subprocess.run(
                    shlex.split(command), stdin=fin, stdout=fout, shell=False,
                    cwd=cwd,
                )

        return proc.returncode, command


def get_calculator(
        calculator: str,
        nproc: int = 1,
        **kwargs: Dict[str, Any],
) -> FileIOCalculator:

    if calculator == 'gamess-us':
        Calc = GAMESSUS
    elif calculator == 'espresso':
        Calc = Espresso

    return Calc(launcher=Launcher(calculator, nproc),
                **kwargs)
