from ase.units import fs, kB, GPa
from ase.build import bulk
from ase.md.nvtberendsen import NVTBerendsen
from ase.md.nptberendsen import NPTBerendsen, Inhomogeneous_NPTBerendsen
from ase.md import MDLogger
from ase.utils import seterr
from ase.md.velocitydistribution import (MaxwellBoltzmannDistribution,
                                             Stationary)
import numpy as np

def test_nvt_berendsen_asap(asap3):
    _berendsen_asap(asap3, None)

def test_npt_berendsen_asap(asap3):
    _berendsen_asap(asap3, 1.0 * GPa)

def test_npt_berendsen_inhom_asap(asap3):
    _berendsen_asap(asap3, 1.0 * GPa, False)


def _berendsen_asap(asap3, pressure, homog=True):
    """Test NVT or NPT Berendsen dynamics.

    The pressure should be in atomic units.
    """
    with seterr(all='raise'):
        rng = np.random.RandomState(None)
        a = bulk('Cu', orthorhombic=True).repeat((6, 6, 6))
        # Introduce an inhomogeneity
        a[7].symbol = 'Au'
        a[8].symbol = 'Au'
        del a[101]
        del a[100]
        print(a)
        a.calc = asap3.EMT()
        # Set temperature to 10 K
        MaxwellBoltzmannDistribution(a, 10 * kB, force_temp=True, rng=rng)
        Stationary(a)
        assert abs(a.get_temperature() - 10) < 0.0001
        # Berendsen dynamics should raise this to 300 K
        T = 300
        if pressure is None:
            md = NVTBerendsen(a, timestep=4 * fs, temperature_K=T,
                                  taut=2000*fs,
                                  logfile='-', loginterval=500)
        elif homog:
            md = NPTBerendsen(a, timestep=4 * fs, temperature_K=T,
                                  taut=2000*fs,
                                  pressure_bar=pressure/GPa*1e4, taup=2000*fs,
                                  compressibility=1 / (140 * 1e4))
            # We want logging with stress included
            md.attach(MDLogger(md, a, '-', stress=True), interval=500)
        else:
            md = Inhomogeneous_NPTBerendsen(
                a, timestep=4 * fs, temperature_K=T, taut=2000*fs,
                pressure_bar=pressure/GPa*1e4, taup=2000*fs,
                compressibility=1 / (140 * 1e4)
                )
            # We want logging with stress included
            md.attach(MDLogger(md, a, '-', stress=True), interval=500)
        md.run(steps=5000)
        # Now gather the temperature over 10000 timesteps, collecting it
        # every 5 steps
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
            
