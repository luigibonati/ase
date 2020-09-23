"""Berendsen NPT dynamics class."""

import numpy as np
import warnings

from ase.md.nvtberendsen import NVTBerendsen
import ase.units as units


class NPTBerendsen(NVTBerendsen):
    """Berendsen (constant N, P, T) molecular dynamics.

    This dynamics scale the velocities and volumes to maintain a constant
    pressure and temperature.  The shape of the simulation cell is not
    altered, if that is desired use Inhomogenous_NPTBerendsen.

    Usage: NPTBerendsen(atoms, timestep, temperature, taut, pressure, taup)

    atoms
        The list of atoms.

    timestep
        The time step.

    temperature
        The desired temperature, in Kelvin.

    taut
        Time constant for Berendsen temperature coupling.

    fixcm
        If True, the position and momentum of the center of mass is
        kept unperturbed.  Default: True.

    pressure
        The desired pressure, in bar (1 bar = 1e5 Pa).

    taup
        Time constant for Berendsen pressure coupling.

    compressibility
        The compressibility of the material, water 4.57E-5 bar-1, in bar-1

    """

    def __init__(self, atoms, timestep, temperature=None,
                 *, temperature_K=None,
                 pressure=None, pressure_au=None, pressure_bar=None,
                 taut=0.5e3 * units.fs, taup=1e3 * units.fs,
                 compressibility=None, fixcm=True, trajectory=None,
                 logfile=None, loginterval=1, append_trajectory=False):

        NVTBerendsen.__init__(self, atoms, timestep, temperature=temperature,
                              temperature_K=temperature_K,
                              taut=taut, fixcm=fixcm, trajectory=trajectory,
                              logfile=logfile, loginterval=loginterval,
                              append_trajectory=append_trajectory)
        self.taup = taup
        self.pressure = self._process_pressure(pressure, pressure_bar,
                                                   pressure_au)
        if compressibility is None:
            raise TypeError("Missing 'compressibility' argument")
        self.set_compressibility(compressibility)

    def set_taup(self, taup):
        self.taup = taup

    def get_taup(self):
        return self.taup

    def set_pressure(self, pressure=None, *, pressure_au=None,
                         pressure_bar=None):
        self.pressure = self._process_pressure(pressure, pressure_bar,
                                                   pressure_au)

    def get_pressure(self):
        return self.pressure

    def set_compressibility(self, compressibility):
        if self.pressure_unit == 'bar':
            self.compressibility = compressibility / (1e5 * units.Pascal)
        else:
            assert self.pressure_unit == 'au'
            self.compressibility = compressibility

    def get_compressibility(self):
        return self.compressibility

    def set_timestep(self, timestep):
        self.dt = timestep

    def get_timestep(self):
        return self.dt

    def scale_positions_and_cell(self):
        """ Do the Berendsen pressure coupling,
        scale the atom position and the simulation cell."""

        taupscl = self.dt / self.taup
        stress = self.atoms.get_stress(voigt=False, include_ideal_gas=True)
        old_pressure = -stress.trace() / 3
        scl_pressure = (1.0 - taupscl * self.compressibility / 3.0 *
                        (self.pressure - old_pressure))

        #print("old_pressure", old_pressure, self.pressure)
        #print("volume scaling by:", scl_pressure)

        cell = self.atoms.get_cell()
        cell = scl_pressure * cell
        self.atoms.set_cell(cell, scale_atoms=True)

    def step(self, f=None):
        """ move one timestep forward using Berenden NPT molecular dynamics."""

        NVTBerendsen.scale_velocities(self)
        self.scale_positions_and_cell()

        #one step velocity verlet
        atoms = self.atoms

        if f is None:
            f = atoms.get_forces()

        p = self.atoms.get_momenta()
        p += 0.5 * self.dt * f

        if self.fixcm:
            # calculate the center of mass
            # momentum and subtract it
            psum = p.sum(axis=0) / float(len(p))
            p = p - psum

        self.atoms.set_positions(
            self.atoms.get_positions() +
            self.dt * p / self.atoms.get_masses()[:, np.newaxis])

        # We need to store the momenta on the atoms before calculating
        # the forces, as in a parallel Asap calculation atoms may
        # migrate during force calculations, and the momenta need to
        # migrate along with the atoms.  For the same reason, we
        # cannot use self.masses in the line above.

        self.atoms.set_momenta(p)
        f = self.atoms.get_forces()
        atoms.set_momenta(self.atoms.get_momenta() + 0.5 * self.dt * f)

        return f

    def _process_pressure(self, pressure, pressure_bar, pressure_au):
        """Handle that pressure can be specified in multiple units.

        For at least a transition period, Berendsen NPT dynamics in ASE can
        have the pressure specified in either bar or atomic units (eV/Å^3).

        Four parameters:

        pressure: None or float
            The original pressure specification in whatever unit was 
            historically used.  A warning is issued if this is not None.

        pressure_bar: None or float
            Pressure in bar.

        pressure_au: None or float
            Pressure in ev/Å^3.

        Exactly one of the three pressure parameters must be different from 
        None, otherwise an error is issued.

        Return value: Pressure in eV/Å^3.
        """
        if ((pressure is not None) + (pressure_bar is not None)
                + (pressure_au is not None)) != 1:
            raise TypeError("Exactly one of the parameters 'pressure',"
                                + " 'pressure_bar', and 'pressure_au' must"
                                + " be given")
        if pressure is not None:
            w = ("Explicitly give the pressure unit by using the"
                     + " 'pressure_bar' or 'pressure_au' argument instead"
                     + " of 'pressure'.")
            warnings.warn(FutureWarning(w))
            pressure_bar = pressure 

        err = "You need to stick to the pressure unit used when creating this object ({})"
        if pressure_bar is not None:
            pressure_au = pressure_bar * (1e5 * units.Pascal)
            if getattr(self, 'pressure_unit', 'bar') != 'bar':
                raise ValueError(err.format(self.pressure_unit))
            self.pressure_unit = 'bar'
        else:
            if getattr(self, 'pressure_unit', 'au') != 'au':
                raise ValueError(err.format(self.pressure_unit))
            self.pressure_unit = 'au'

        return pressure_au


    


class Inhomogeneous_NPTBerendsen(NPTBerendsen):
    """Berendsen (constant N, P, T) molecular dynamics.

    This dynamics scale the velocities and volumes to maintain a constant
    pressure and temperature.  The size of the unit cell is allowed to change
    independently in the three directions, but the angles remain constant.

    Usage: NPTBerendsen(atoms, timestep, temperature, taut, pressure, taup)

    atoms
        The list of atoms.

    timestep
        The time step.

    temperature
        The desired temperature, in Kelvin.

    taut
        Time constant for Berendsen temperature coupling.

    fixcm
        If True, the position and momentum of the center of mass is
        kept unperturbed.  Default: True.

    pressure
        The desired pressure, in bar (1 bar = 1e5 Pa).

    taup
        Time constant for Berendsen pressure coupling.

    compressibility
        The compressibility of the material, water 4.57E-5 bar-1, in bar-1

    mask
        Specifies which axes participate in the barostat.  Default (1, 1, 1)
        means that all axes participate, set any of them to zero to disable
        the barostat in that direction.
    """
    def __init__(self, atoms, timestep, temperature=None,
                 *, temperature_K=None, 
                 taut=0.5e3 * units.fs, pressure=None,
                 pressure_bar=None, pressure_au=None, taup=1e3 * units.fs,
                 compressibility=None, mask=(1, 1, 1),
                 fixcm=True, trajectory=None,
                 logfile=None, loginterval=1):

        NPTBerendsen.__init__(self, atoms, timestep, temperature=temperature,
                              temperature_K=temperature_K,
                              taut=taut, taup=taup, pressure=pressure,
                              pressure_bar=pressure_bar,
                              pressure_au=pressure_au,
                              compressibility=compressibility,
                              fixcm=fixcm, trajectory=trajectory,
                              logfile=logfile, loginterval=loginterval)
        self.mask = mask

    def scale_positions_and_cell(self):
        """ Do the Berendsen pressure coupling,
        scale the atom position and the simulation cell."""

        taupscl = self.dt * self.compressibility / self.taup / 3.0
        stress = - self.atoms.get_stress(include_ideal_gas=True)
        if stress.shape == (6,):
            stress = stress[:3]
        elif stress.shape == (3, 3):
            stress = [stress[i][i] for i in range(3)]
        else:
            raise ValueError('Cannot use a stress tensor of shape ' +
                             str(stress.shape))
        pbc = self.atoms.get_pbc()
        scl_pressurex = 1.0 - taupscl * (self.pressure - stress[0]) \
                        * pbc[0] * self.mask[0]
        scl_pressurey = 1.0 - taupscl * (self.pressure - stress[1]) \
                        * pbc[1] * self.mask[1]
        scl_pressurez = 1.0 - taupscl * (self.pressure - stress[2]) \
                        * pbc[2] * self.mask[2]
        cell = self.atoms.get_cell()
        cell = np.array([scl_pressurex * cell[0],
                         scl_pressurey * cell[1],
                         scl_pressurez * cell[2]])
        self.atoms.set_cell(cell, scale_atoms=True)
