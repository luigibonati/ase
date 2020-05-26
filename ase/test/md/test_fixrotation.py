from ase.units import fs, kB
from ase.build import bulk
from ase.md import Langevin
from ase.md.fix import FixRotation
from ase.utils import seterr
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
import numpy as np


def check_inertia(atoms):
    m, v = atoms.get_moments_of_inertia(vectors=True)
    print("Moments of inertia:")
    print(m)
    # There should be one vector in the z-direction
    n = 0
    delta = 1e-2
    for a in v:
        if (abs(a[0]) < delta
            and abs(a[1]) < delta
            and abs(abs(a[2]) - 1.0) < delta):

            print("Vector along z:", a)
            n += 1
        else:
            print("Vector not along z:", a)
    assert n == 1


def test_verlet_asap(asap3):
    with seterr(all='raise'):
        a = bulk('Au', cubic=True).repeat((3, 3, 20))
        a.pbc = False
        a.center(vacuum=5.0 + np.max(a.cell) / 2)
        print(a)
        a.calc = asap3.EMT()
        MaxwellBoltzmannDistribution(a, 300 * kB, force_temp=True)
        Stationary(a)
        check_inertia(a)
        md = Langevin(
            a,
            timestep=5 * fs,
            temperature=300 * kB,
            friction=1e-3,
            logfile='-',
            loginterval=500)
        fx = FixRotation(a)
        md.attach(fx)
        md.run(steps=10000)
        check_inertia(a)
