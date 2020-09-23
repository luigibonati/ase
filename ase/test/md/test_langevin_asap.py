from ase.units import fs, kB
from ase.build import bulk
from ase.md import Langevin
from ase.utils import seterr
from ase.md.velocitydistribution import (MaxwellBoltzmannDistribution,
                                             Stationary)
import numpy as np

def test_langevin_asap(asap3):
    with seterr(all='raise'):
        rng = np.random.RandomState(0)
        a = bulk('Au', cubic=True).repeat((5, 5, 5))
        a.pbc = (False, False, False)
        a.center(vacuum=2.0)
        print(a)
        a.calc = asap3.EMT()
        # Set temperature to 10 K
        MaxwellBoltzmannDistribution(a, 10 * kB, force_temp=True, rng=rng)
        Stationary(a)
        assert abs(a.get_temperature() - 10) < 0.0001
        # Langevin dynamics should raise this to 300 K
        T = 300
        md = Langevin(a, timestep=4 * fs, temperature_K=T, friction=0.01,
                          logfile='-', loginterval=500, rng=rng)
        md.run(steps=5000)
        # Now gather the temperature over 10000 timesteps, collecting it
        # every 5 steps
        temp = []
        energy = []
        for i in range(2000):
            md.run(steps=5)
            temp.append(a.get_temperature())
            energy.append(a.get_potential_energy() + a.get_kinetic_energy())
        temp = np.array(temp)
        avgtemp = np.mean(temp)
        fluct = np.std(temp)
        avgenergy = np.mean(energy)
        print("Temperature is {:.2f} K +/- {:.2f} K".format(avgtemp, fluct))
        assert abs(avgtemp - T) < 10.0
        # Calculate the heat capacity, should be close to 3*kB per atom
        dT = 25
        md.set_temperature(temperature_K=T + dT)
        md.run(steps=2000)
        temp = []
        energy = []
        for i in range(5000):
            md.run(steps=5)
            temp.append(a.get_temperature())
            energy.append(a.get_potential_energy() + a.get_kinetic_energy())
        temp = np.array(temp)
        avgtemp = np.mean(temp)
        fluct = np.std(temp)
        avgenergy2 = np.mean(energy)
        print("Temperature is {:.2f} K +/- {:.2f} K".format(avgtemp, fluct))
        assert abs(avgtemp - (T + dT)) < 10.0
        cv = (avgenergy2 - avgenergy) / (len(a) * kB * dT)
        print("Heat capacity per atom: {:.3f} kB".format(cv))
        # We need much longer simulations for good statistics, hence the large
        # allowed error.
        assert abs(cv - 3.0) < 1.2 
