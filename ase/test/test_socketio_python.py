import sys
import numpy as np
from ase.calculators.socketio import SocketIOCalculator
from ase.calculators.emt import EMT


def launch_client(atoms, properties=None, port=None, unixsocket=None):
    import pickle
    from subprocess import Popen, PIPE
    proc = Popen([sys.executable, '-m', 'ase.calculators.socketio'],
                 stdin=PIPE)

    pickle.dump(dict(unixsocket=unixsocket, port=port), proc.stdin)
    pickle.dump(atoms.copy(), proc.stdin)
    pickle.dump(get_calculator, proc.stdin)
    proc.stdin.close()
    return proc


def get_calculator():
    return EMT()


def test_socketio_python():
    from ase.build import bulk
    from ase.constraints import ExpCellFilter
    from ase.optimize import BFGS

    atoms = bulk('Au') * (2, 2, 2)
    atoms.rattle(stdev=0.05)
    fmax = 0.01
    atoms.cell += np.random.RandomState(42).rand(3, 3) * 0.05
    with SocketIOCalculator(launch_client=launch_client,
                            unixsocket='ase-socketio') as calc:
        atoms.calc = calc
        opt = BFGS(ExpCellFilter(atoms))
        opt.run(fmax=fmax)
    forces = atoms.get_forces()
    assert np.linalg.norm(forces, axis=0).max() < fmax
