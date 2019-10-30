import sys
from subprocess import Popen
from ase.calculators.calculator import Calculator
import pickle

class SubProcessCalculator(Calculator):
    def __init__(self, calc):
        self.calc = calc
        self.proc = None

    @property
    def is_open(self):
        return self.proc is None

    def __enter__(self):
        self.proc = Popen([sys.executable, '-m', __name__],
                          stdout=PIPE, stdin=PIPE, stderr=PIPE)
        ...

    def __exit__(self, *args):
        self.proc.wait()
        self.proc = None

    def calculate(self, atoms, properties=None, system_changes=None):
        pickle.dump(self.calc, atoms, properties, system_changes)
        exitcode = proc.wait()
        if exitcode:
            raise OSError()


def main():
    while True:
        calc, atoms, properties, system_changes = pickle.load(sys.stdin)
        calc.calculate....

if __name__ == '__main__':
    main()
