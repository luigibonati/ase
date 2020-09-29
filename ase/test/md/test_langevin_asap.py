from ase.units import fs
from ase.build import bulk
from ase.md import Langevin
from ase.utils import seterr
from ase.md.velocitydistribution import (MaxwellBoltzmannDistribution,
                                         Stationary)
import numpy as np
import pytest

@pytest.mark.slow
def test_langevin_asap(asap3):
    with seterr(all='raise'):
        rng = np.random.RandomState(0)
        a = bulk('Au', cubic=True).repeat((4, 4, 4))
        a.pbc = (False, False, False)
        a.center(vacuum=2.0)
        print(a)
        a.calc = asap3.EMT()
        # Set temperature to 10 K
        MaxwellBoltzmannDistribution(
            a, temperature_K=10, force_temp=True, rng=rng)
        Stationary(a)
        assert abs(a.get_temperature() - 10) < 0.0001
        # Langevin dynamics should raise this to 300 K
        T = 300
        md = Langevin(a, timestep=4 * fs, temperature_K=T, friction=0.01,
                      logfile='-', loginterval=500, rng=rng)
        md.run(steps=3000)
        # Now gather the temperature over 10000 timesteps, collecting it
        # every 5 steps
        temp = []
        for i in range(1500):
            md.run(steps=5)
            temp.append(a.get_temperature())
        temp = np.array(temp)
        avgtemp = np.mean(temp)
        fluct = np.std(temp)
        print("Temperature is {:.2f} K +/- {:.2f} K".format(avgtemp, fluct))
        assert abs(avgtemp - T) < 10.0
