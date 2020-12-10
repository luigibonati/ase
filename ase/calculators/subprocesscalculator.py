import sys
from abc import ABC, abstractmethod
import pickle
from subprocess import Popen, PIPE
from ase.calculators.calculator import Calculator, all_properties
from ase.calculators.singlepoint import SinglePointDFTCalculator


class PackedCalculator(ABC):
    """Not all calculators can be pickled or otherwise packed."""

    @abstractmethod
    def unpack_calculator(self) -> Calculator:
        """Return the calculator packed inside.

        This is called on the process doing the actual calculation
        to create the calculator that does the real work."""

    def calculator(self) -> 'SubProcessPythonCalculator':
        """Return a SubProcessPythonCalculator for this calculator.

        The subprocess calculator wraps a subprocess containing
        the actual calculator, and computations are done inside that
        subprocess."""
        return PythonSubProcessCalculator(self)


class NamedPackedCalculator(PackedCalculator):
    def __init__(self, name, kwargs=None):
        self._name = name
        if kwargs is None:
            kwargs = {}
        self._kwargs = kwargs

    def unpack_calculator(self):
        from ase.calculators.calculator import get_calculator_class
        cls = get_calculator_class(self._name)
        return cls(**self._kwargs)

    def __repr__(self):
        return 'CalculatorInput({}, {})'.format(self._name, self._kwargs)


class PythonSubProcessCalculator(Calculator):
    """Calculator for running calculations in external processes.

    TODO: This should work with arbitrary commands including MPI stuff.

    This calculator runs a subprocess wherein it sets up an
    actual calculator.  Calculations are forwarded through pickle
    to that calculator, which returns results through pickle."""
    implemented_properties = all_properties
    def __init__(self, calc_input):
        super().__init__()
        proc = Popen([sys.executable, '-m', __name__],
                     stdout=PIPE, stdin=PIPE)
        self.proc = proc
        self.calc_input = calc_input

        self._send(calc_input)

    def set(self, **kwargs):
        if hasattr(self, 'proc'):
            raise RuntimeError('No setting things for now, thanks')

    def _send(self, obj):
        pickle.dump(obj, self.proc.stdin)
        self.proc.stdin.flush()

    def _recv(self):
        return pickle.load(self.proc.stdout)

    def __repr__(self):
        return '{}({})'.format(type(self).__name__,
                               self.calc_input)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self._send('stop')
        self.proc.wait()
        self.proc = None

    def _run_calculation(self, atoms, properties, system_changes):
        self._send('calculate')
        self._send((atoms, properties, system_changes))

    def calculate(self, atoms, properties, system_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        # We send a pickle of self.atoms because this is a fresh copy
        # of the input, but without an unpicklable calculator:
        self._run_calculation(self.atoms.copy(), properties, system_changes)
        results = self._recv()
        self.results.update(results)


def main():
    # We switch stdout so stray print statements won't interfere with outputs:
    binary_stdout = sys.stdout.buffer
    sys.stdout = sys.stderr

    def recv():
        return pickle.load(sys.stdin.buffer)

    def send(obj):
        pickle.dump(obj, binary_stdout)
        binary_stdout.flush()

    pack = recv()
    calc = pack.unpack_calculator()

    while True:
        instruction = recv()
        if instruction == 'stop':
            break
        elif instruction != 'calculate':
            raise ValueError('Bad instruction: {}'.format(instruction))

        atoms, properties, system_changes = recv()

        # Again we need formalization of the results/outputs, and
        # a way to programmatically access all available properties.
        # We do a wild hack for now:
        calc.results.clear()
        # If we don't clear(), the caching is broken!  For stress.
        # But not for forces.  What dark magic from the depths of the
        # underworld is at play here?
        calc.calculate(atoms=atoms, properties=properties,
                       system_changes=system_changes)
        results = calc.results
        send(results)


if __name__ == '__main__':
    main()
