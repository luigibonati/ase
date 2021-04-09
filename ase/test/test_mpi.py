import sys
from subprocess import run

import numpy as np
from ase.parallel import world
from ase.test.mpi import run_in_parallel


def test_mpi_unused_on_import():
    """Try to import all ASE modules and check that ase.parallel.world has not
    been used.  We want to delay use of world until after MPI4PY has been
    imported.

    We run the test in a subprocess so that we have a clean Python interpreter.
    """

    # Should cover most of ASE:
    modules = ['ase.optimize',
               'ase.db',
               'ase.gui']

    imports = 'import ' + ', '.join(modules)

    run([sys.executable,
         '-c',
         '{imports}; from ase.parallel import world; assert world.comm is None'
         .format(imports=imports)],
        check=True)


@run_in_parallel(size=2)
def test_mpi():
    print(world.rank)
    a = np.ones(3)
    world.sum(a)
    print(a)
    world.sum(a, 0)
    print(a)
    world.broadcast(a, 0)
    print(a)
    print(world.rank, world.sum(1))
    print(world.rank, world.sum(1.5))
    print(world.rank, world.sum(1, 0))
    print(world.rank, world.sum(1.5, 1))
