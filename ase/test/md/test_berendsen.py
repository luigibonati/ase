import pytest
from ase import Atoms
from ase.units import fs, kB, GPa
from ase.build import bulk
from ase.md.nvtberendsen import NVTBerendsen
from ase.md.nptberendsen import NPTBerendsen
from ase.io import Trajectory, read
from ase.optimize import QuasiNewton
from ase.utils import seterr
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
import numpy as np

def propagate(asap3, algorithm, **kwargs):
    with seterr(all='raise'):
        a = bulk('Au').repeat((5,5,5))   # Must be big enough to avoid ridiculous fluctuations
        #a[5].symbol = 'Ag'
        a.pbc = (True, True, False)
        print(a)
        a.calc = asap3.EMT()
        MaxwellBoltzmannDistribution(a, 100*kB, force_temp=True)
        Stationary(a)
        assert abs(a.get_temperature() - 100) < 0.0001
        md = algorithm(a, timestep=2 * fs, logfile='-', loginterval=1000, **kwargs)
        e0 = a.get_total_energy()
        # Equilibrate for 20 ps
        md.run(steps=10000)
        # Gather data for 100 ps
        T = []
        p = []
        for i in range(10000):
            md.run(5)
            T.append(a.get_temperature())
            pres = - a.get_stress(include_ideal_gas=True)[:3].sum() / 3
            p.append(pres)
        Tmean = np.mean(T)
        p = np.array(p) / GPa * 1000
        pmean = np.mean(p)
        print('Temperature: {:.2f}K +/- {:.2f}K  (N={})'.format(Tmean, np.std(T), len(T)))
        print('Pressure: {:.2f}MPa +/- {:.2f}MPa  (N={})'.format(pmean, np.std(p), len(p)))
        return Tmean, pmean, a


def test_nvtberendsen(asap3):
    t, _, _ = propagate(asap3, NVTBerendsen, temperature = 300, taut=1000*fs)
    assert abs(t - 300) < 1.0

def test_nptberendsen(asap3):
    Bgold = 220.0 * 10000  # Bulk modulus of gold, in bar (1 GPa = 10000 bar)
    t, p, _ = propagate(asap3, NPTBerendsen, temperature = 300, pressure=5000, taut=1000*fs, taup=1000*fs,
                        compressibility=1/Bgold)
    assert abs(t - 300) < 2.0
    assert abs(p - 500) < 10.0
