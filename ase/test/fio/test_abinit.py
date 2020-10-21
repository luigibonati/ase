from ase.io import read, write
from ase.build import bulk
from ase.calculators.calculator import compare_atoms


def test_abinit_roundtrip():
    m1 = bulk('Ti')
    m1.set_initial_magnetic_moments(range(len(m1)))
    write('abinit_save.in', images=m1, format='abinit-in')
    m2 = read('abinit_save.in', format='abinit-in')

    # (How many decimals?)
    assert not compare_atoms(m1, m2, tol=1e-7)
