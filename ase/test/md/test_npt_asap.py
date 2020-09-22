from ase.units import fs, kB, GPa
from ase.build import bulk
from ase.md.npt import NPT
from ase.md.nptberendsen import NPTBerendsen
from ase.md import MDLogger
from ase.utils import seterr
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
import numpy as np


def test_npt_asap(asap3):
    with seterr(all='raise'):
        pressure = 1.0 * GPa
        rng = np.random.RandomState(0)
        a = bulk('Cu', orthorhombic=True).repeat((6, 6, 6))
        print(a)
        a.calc = asap3.EMT()
        # NPT Dynamics is awful at setting correct pressure and
        # temperature, but should be good at maintaining it.  Use
        # NPTBerendsen to hit the right value.
        T = 300
        MaxwellBoltzmannDistribution(a, T * kB, force_temp=True, rng=rng)
        Stationary(a)
        berend = NPTBerendsen(a, timestep=4 * fs, temperature=T, taut=2000*fs,
                              pressure=pressure/GPa*1e4, taup=2000*fs,
                              compressibility=1 / (140 * 1e4),
                              logfile='-', loginterval=500)
        berend.run(steps=5000)
        # Now gather the temperature over 10000 timesteps, collecting it every 5 steps
        ptime = 2000 * fs
        md = NPT(a, timestep=4 * fs, temperature=T*kB, externalstress=pressure,
                     ttime=2000 * fs, pfactor=ptime**2/(140 / GPa))
        # We want logging with stress included
        md.attach(MDLogger(md, a, '-', stress=True), interval=500)
        temp = []
        press = []
        for i in range(2000):
            md.run(steps=5)
            temp.append(a.get_temperature())
            p = -a.get_stress(include_ideal_gas=True)[:3].sum() / 3.0
            press.append(p)
        temp = np.array(temp)
        avgtemp = np.mean(temp)
        fluct = np.std(temp)
        avgpressure = np.mean(press)
        print("Temperature is {:.2f} K +/- {:.2f} K.".format(avgtemp, fluct))
        print("Pressure is {:.4f} GPa.".format(avgpressure / GPa))
        assert abs(avgtemp - T) < 10.0
        if pressure is not None:
            assert abs(avgpressure - pressure) < 0.02 * GPa
            
