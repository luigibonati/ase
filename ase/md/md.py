"""Molecular Dynamics."""

import warnings
import numpy as np

from ase.optimize.optimize import Dynamics
from ase.md.logger import MDLogger
from ase.io.trajectory import Trajectory
from ase import units


class MolecularDynamics(Dynamics):
    """Base-class for all MD classes."""
    def __init__(self, atoms, timestep, trajectory, logfile=None,
                 loginterval=1, append_trajectory=False):

        # dt as to be attached _before_ parent class is initialized
        self.dt = timestep

        Dynamics.__init__(self, atoms, logfile=None, trajectory=None)

        self.masses = self.atoms.get_masses()
        self.max_steps = None

        if 0 in self.masses:
            warnings.warn('Zero mass encountered in atoms; this will '
                          'likely lead to errors if the massless atoms '
                          'are unconstrained.')

        self.masses.shape = (-1, 1)

        if not self.atoms.has('momenta'):
            self.atoms.set_momenta(np.zeros([len(self.atoms), 3]))

        # Trajectory is attached here instead of in Dynamics.__init__
        # to respect the loginterval argument.
        if trajectory is not None:
            if isinstance(trajectory, str):
                mode = "a" if append_trajectory else "w"
                trajectory = Trajectory(trajectory, mode=mode, atoms=atoms)
            self.attach(trajectory, interval=loginterval)

        if logfile:
            self.attach(MDLogger(dyn=self, atoms=atoms, logfile=logfile),
                        interval=loginterval)

    def todict(self):
        return {'type': 'molecular-dynamics',
                'md-type': self.__class__.__name__,
                'timestep': self.dt}

    def irun(self, steps=50):
        """ Call Dynamics.irun and adjust max_steps """
        self.max_steps = steps + self.nsteps
        return Dynamics.irun(self)

    def run(self, steps=50):
        """ Call Dynamics.run and adjust max_steps """
        self.max_steps = steps + self.nsteps
        return Dynamics.run(self)

    def get_time(self):
        return self.nsteps * self.dt

    def converged(self):
        """ MD is 'converged' when number of maximum steps is reached. """
        return self.nsteps >= self.max_steps

    def _process_temperature(self, temperature, temperature_K, temperature_eV, orig_unit):
        """Handle that temperature can be specified in multiple units.

        For at least a transition period, molecular dynamics in ASE can
        have the temperature specified in either Kelvin or Electron
        Volt.  The different MD algorithms had different defaults, by
        forcing the user to explicitly choose a unit we can resolve
        this.  Using the original method then will issue a
        FutureWarning.

        Four parameters:

        temperature: None or float
            The original temperature specification in whatever unit was historically used.
            A warning is issued if this is not None.

        temperature_K: None or float
            Temperature in Kelvin.

        temperature_eV: None or float
            Temperature in eV.

        orig_unit: str
            Unit used for the `temperature`` parameter.  Must be 'K' or 'eV'.

        Exactly one of the three temperature parameters must be different from None,
        otherwise an error is issued.

        Return value: Temperature in eV.
        """
        if ((temperature is not None) + (temperature_K is not None)
                + (temperature_eV is not None)) != 1:
            raise TypeError("Exactly one of the parameters 'temperature',"
                                + " 'temperature_K', and 'temperature_eV' must"
                                + " be given")
        if temperature is not None:
            w = ("Explicitly give the temperature unit by using the"
                     + " 'temperature_K' or 'temperature_eV' argument instead"
                     + " of 'temperature'.")
            warnings.warn(FutureWarning(w))
            if orig_unit == 'K':
                return temperature * units.kB
            elif orig_unit == 'eV':
                return temperature
            else:
                raise ValueError("Unknown temperature unit "+orig_unit)

        if temperature_K is not None:
            return temperature_K * units.kB

        assert temperature_eV is not None
        return temperature_eV

