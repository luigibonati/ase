import sys
from subprocess import Popen
import numpy as np
from ase.calculators.socketio import SocketIOCalculator
from ase.calculators.emt import EMT


class EMTSocketClient:
    def __init__(self):
        self._calc = EMT()
        self.command = f'{sys.executable} -m ase.calculators.socketio'
        self.prefix = 'emt'
        self.directory = '.'

    def write_input(self, *args, **kwargs):
        pass


def test_socketio_python():
    from ase.build import bulk
    from ase.constraints import ExpCellFilter
    from ase.optimize import BFGS
    clientcalc = EMTSocketClient()

    atoms = bulk('Au') * (2, 2, 2)
    atoms.rattle(stdev=0.1)
    fmax = 0.01
    atoms.cell += np.random.RandomState(42).rand(3, 3) * 0.1
    with SocketIOCalculator(clientcalc, unixsocket='ase-socketio') as calc:
        atoms.calc = calc
        opt = BFGS(ExpCellFilter(atoms))
        opt.run(fmax=fmax)
    forces = atoms.get_forces()
    assert np.linalg.norm(forces, axis=0).max() < fmax
