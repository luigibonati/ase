import sys
from subprocess import Popen, PIPE
from ase.calculators.calculator import Calculator, all_properties
from ase.calculators.singlepoint import SinglePointDFTCalculator
import pickle


class CalculatorInput:
    def __init__(self, name, kwargs):
        self.name = name
        self.kwargs = kwargs

    def __repr__(self):
        return 'CalculatorInput({}, {})'.format(self.name, self.kwargs)


def wrap_subprocess(calc):
    calc_input = CalculatorInput(calc.name, calc.parameters)
    return SubProcessPythonCalculator(calc_input)

# TODO: Generalize outputs somehow?
#class CalculatorOutput:
#    def __init__(self, results):
#        self.results = results

class SubProcessPythonCalculator(Calculator):
    """Calculator for running calculations in external processes.

    TODO: This should work with arbitrary commands including MPI stuff.

    This calculator runs a subprocess wherein it sets up an
    actual calculator.  Calculations are forwarded through pickle
    to that calculator, which returns results through pickle."""
    implemented_properties = all_properties
    def __init__(self, calc_input):
        super().__init__()
        proc = Popen([sys.executable, '-m', __name__],
                     stdout=PIPE, stdin=PIPE)  #, stderr=PIPE)
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
        self._run_calculation(self.atoms, properties, system_changes)
        results = self._recv()
        self.results.update(results)


def main():
    from ase.calculators.calculator import get_calculator_class

    # We switch stdout so stray print statements won't interfere with outputs:
    binary_stdout = sys.stdout.buffer
    sys.stdout = sys.stderr

    def recv():
        return pickle.load(sys.stdin.buffer)

    def send(obj):
        pickle.dump(obj, binary_stdout)
        binary_stdout.flush()

    calc_input = recv()
    cls = get_calculator_class(calc_input.name)
    calc = cls(**calc_input.kwargs)

    while True:
        instruction = recv()
        if instruction == 'stop':
            break
        elif instruction != 'calculate':
            raise ValueError('Bad instruction: {}'.format(instruction))

        atoms, properties, system_changes = recv()

        calc.calculate(atoms=atoms, properties=properties,
                       system_changes=system_changes)
        results = calc.results
        send(results)

if __name__ == '__main__':
    main()
