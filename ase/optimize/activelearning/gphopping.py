import numpy as np
import time
import copy
import os
from ase.calculators.gp.calculator import GPCalculator
from ase import io, units
from ase.parallel import parprint, parallel_function
from ase.optimize.activelearning.acquisition import acquisition
from ase.md import VelocityVerlet
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.optimize import QuasiNewton
from scipy.signal import find_peaks
from ase.optimize.minimahopping import ComparePositions


class GPHopping:

    def __init__(self, atoms, calculator, model_calculator=None,
                 force_consistent=None, optimizer=QuasiNewton,
                 trajectory='GPHopping.traj', T0=500., beta1=1.01, beta2=0.99,
                 energy_threshold=10., geometry_threshold=0.5, mdmin=4,
                 timestep=1.0, maxtime=1000., maxstep=0.5, maxoptsteps=300,
                 hop_strategy='lowest_energy_minimum'):
        """
        Parameters
        --------------
        atoms: Atoms object
            The Atoms object to relax.

        model_calculator: Model object.
            Model calculator to be used for predicting the potential energy
            surface. The default is None which uses a GP model with the Squared
            Exponential Kernel and other default parameters. See
            *ase.calculator.gp.calculator* GPModel for default GP parameters.

        force_consistent: boolean or None
            Use force-consistent energy calls (as opposed to the energy
            extrapolated to 0 K). By default (force_consistent=None) uses
            force-consistent energies if available in the calculator, but
            falls back on force_consistent=False if not.

                trajectory: string
            Filename to store the predicted optimization.
                Additional information:
                - Energy uncertain: The energy uncertainty in each image can be
                  accessed in image.info['uncertainty'].

        """

        # MD simulation parameters.
        self.T0 = T0
        self.energy_threshold = energy_threshold
        self.mdmin = mdmin
        self.timestep = timestep
        self.maxtime = maxtime
        self.maxstep = maxstep
        self.maxoptsteps = maxoptsteps

        # Hopping parameters.
        self.beta1 = beta1  # Increase temperature.
        self.beta2 = beta2  # Decrease temperature.
        self.geometry_threshold = geometry_threshold
        self.optimizer = optimizer
        self.hop_strategy = hop_strategy

        # Model parameters.
        self.model_calculator = model_calculator
        # Default GP Calculator parameters if not specified by the user.
        if model_calculator is None:
            self.model_calculator = GPCalculator(train_images=[],
                                                 wrap_positions=True)
        self.model_calculator.calculate_uncertainty = True

        # Active Learning setup (single-point calculations).
        self.function_calls = 0
        self.force_calls = 0
        self.ase_calc = calculator
        self.atoms = atoms

        self.constraints = self.atoms.constraints
        self.force_consistent = force_consistent
        self.trajectory = trajectory

    def run(self, fmax=0.05, steps=3000):

        """
        Executing run will start the optimization process.

        Parameters
        ----------
        fmax : float
            Convergence criteria (in eV/Angstrom).

        steps: int
            Maximum number of steps for the surrogate.

        """
        trajectory_main = self.trajectory.split('.')[0]
        trajectory_observations = trajectory_main + '_observations.traj'
        trajectory_candidates = trajectory_main + '_candidates.traj'
        trajectory_minima = trajectory_main + '_found_minima.traj'

        self.atoms.get_potential_energy(force_consistent=self.force_consistent)
        self.atoms.get_forces()
        dump_trajectory(atoms=self.atoms,
                        filename=trajectory_observations,
                        restart=True)
        train_images = io.read(trajectory_observations, ':')

        # Dump to trajectory_minima file the observations that satisfy fmax.
        if os.path.exists(trajectory_minima):
            os.remove(trajectory_minima)
        for i in train_images:
            if get_fmax(i) <= fmax:
                dump_trajectory(atoms=i,
                                filename=trajectory_minima,
                                restart=True)

        temperature = self.T0  # Initial temperature.

        while self.function_calls <= steps:
            # 1. Collect observations.
            train_images = io.read(trajectory_observations, ':')
            # 2. Update GP calculator.
            gp_calc = copy.deepcopy(self.model_calculator)
            gp_calc.update_train_data(train_images=train_images)

            prev_minima = io.read(trajectory_minima, ':')

            if self.hop_strategy == 'lowest_energy_minimum':
                e_prev = []
                for i in prev_minima:
                    e_prev.append(i.get_potential_energy(force_consistent=self.force_consistent))
                parprint('Starting from prev. minima number: ', np.argmin(e_prev))
                self.atoms = prev_minima[np.argmin(e_prev)]

            if self.hop_strategy == 'last_minimum':
                parprint('Starting from last minima found...')
                self.atoms = io.read(trajectory_minima, -1)

            if self.hop_strategy == 'initial_minimum':
                parprint('Starting from last minima found...')
                self.atoms = io.read(trajectory_minima, 0)

            # 3. Perform MD simulation in the predicted PES.
            self.atoms.set_calculator(gp_calc)
            self.atoms.get_potential_energy(force_consistent=self.force_consistent)

            # 3.1 Optimize output from MD simulation to find candidates.

            candidates = []
            np.random.seed(self.function_calls)

            while len(candidates) < 1:
                md_atoms = copy.deepcopy(self.atoms)
                md_results = mdsim(atoms=md_atoms,
                                   temperature=temperature,
                                   maxstep=self.maxstep,
                                   timestep=self.timestep,
                                   maxtime=self.maxtime,
                                   mdmin=self.mdmin,
                                   energy_threshold=self.energy_threshold,
                                   trajectory='md_simulation.traj',
                                   force_consistent=self.force_consistent)

                parprint('Stopping reason:', md_results['stop_reason'])

                for index in md_results['indexes_minima']:
                    atoms = copy.deepcopy(self.atoms)
                    atoms.positions = md_results['positions'][index]
                    candidates += [atoms]

                if md_results['stop_reason'] == 'max_time_reached':
                    parprint('Increasing temp. due to max. time reached.')
                    temperature *= self.beta1  # Increase temperature.

                if md_results['stop_reason'] == 'max_energy_reached':
                    parprint('Decreasing temp. due to max. energy reached.')
                    temperature *= self.beta2  # Decrease temperature.

                # Optimize candidates.

                optimized_candidates = []
                for i in range(0, len(candidates)):
                    atoms = copy.deepcopy(candidates[i])
                    atoms.get_potential_energy(force_consistent=self.force_consistent)
                    atoms = optimize_atoms(atoms=atoms, maxstep=self.maxstep,
                                           fmax=fmax*0.01,
                                           optimizer=self.optimizer,
                                           maxoptsteps=self.maxoptsteps,
                                           force_consistent=self.force_consistent)
                    optimized_candidates += [copy.deepcopy(atoms)]
                candidates = copy.deepcopy(optimized_candidates)

                # Remove candidates that are close to the other found minima.
                filtered_candidates = []
                for i in candidates:
                    unique_candidate = True
                    for j in prev_minima:
                        compare = ComparePositions(translate=True)
                        dmax = compare(atoms1=i, atoms2=j)
                        if dmax <= self.geometry_threshold:
                            unique_candidate = False
                            parprint('Re-found minima. Increasing temperature')
                            temperature *= self.beta1  # Increase temperature.
                    if unique_candidate is True:
                        filtered_candidates += [copy.deepcopy(i)]
                candidates = copy.deepcopy(filtered_candidates)

                if md_results['stop_reason'] == 'max_uncertainty_reached':
                    atoms = copy.deepcopy(self.atoms)
                    atoms.positions = md_results['positions'][-1]
                    candidates += [atoms]
                    temperature *= self.beta2  # Decrease temperature.

                # No candidates? Increase temperature.
                if len(candidates) == 0:
                    temperature *= self.beta1  # Increase temperature.

                parprint('Current temperature:', temperature)

            # 4. Order candidates using acquisition function:
            sorted_candidates = acquisition(train_images=train_images,
                                            candidates=candidates,
                                            mode='energy',
                                            objective='min')

            # Select the best candidate.
            best_candidate = sorted_candidates.pop(0)

            # Save the other candidates for multi-task optimization.
            io.write(trajectory_candidates, sorted_candidates)

            # 5. Evaluate the target function and save it in *observations*.
            self.atoms.positions = best_candidate.get_positions()
            self.atoms.set_calculator(self.ase_calc)
            self.atoms.get_potential_energy(force_consistent=self.force_consistent)
            self.atoms.get_forces()
            dump_trajectory(atoms=self.atoms,
                            filename=trajectory_observations,
                            restart=True)
            self.function_calls += 1
            self.force_calls += 1

            # If the last evaluated structure is below fmax:
            if get_fmax(self.atoms) <= fmax:
                dump_trajectory(atoms=self.atoms,
                                filename=trajectory_minima, restart=True)
            # 6. Print output.
            msg = "--------------------------------------------------------"
            parprint(msg)
            parprint('Step:', self.function_calls)
            parprint('Time:', time.strftime("%m/%d/%Y, %H:%M:%S",
                                            time.localtime()))
            parprint('Energy', self.atoms.get_potential_energy(self.force_consistent))
            parprint("fmax:", get_fmax(self.atoms))
            parprint("Minima found:", len(io.read(trajectory_minima, ':')))
            msg = "--------------------------------------------------------\n"
            parprint(msg)


@parallel_function
def optimize_atoms(atoms, optimizer, fmax, maxstep=0.4, maxoptsteps=1000,
                   trajectory=None, logfile=None, force_consistent=None):
    converged = False
    step = 0
    while converged is False:
        if get_fmax(atoms) <= fmax or step > maxoptsteps or \
                atoms.get_calculator().results['uncertainty'] >= maxstep:
            converged = True
        else:
            opt = optimizer(atoms=atoms, logfile=logfile,
            trajectory=trajectory, force_consistent=force_consistent)
            opt.run(fmax=fmax, steps=1)
            step += 1
    return atoms


@parallel_function
def mdsim(atoms, temperature, maxstep=0.4, timestep=1.0, maxtime=2000.,
          mdmin=1, energy_threshold=2.0, trajectory='md_simulation.traj',
          force_consistent=None):
    """Performs a molecular dynamics simulation until any of the following
    conditions is achieved: (A) Max uncertainty reached (determined by
    'max_step') or (B) 'mdmin' number of minima are found or (C)
    'max_time' (in fs) has been reached or (D) energy threshold has been
    crossed over ('energy_threshold')."""

    initial_energy = atoms.get_potential_energy(force_consistent=force_consistent)  # Initial energy.
    current_time = 0.0  # Initial time.
    energies, uncertainties, positions, indexes_minima = [], [], [], []
    stop_reason = None

    MaxwellBoltzmannDistribution(atoms,
                                 temp=temperature * units.kB,
                                 force_temp=True)
    dyn = VelocityVerlet(atoms, timestep=timestep * units.fs,
                         trajectory=trajectory)

    while stop_reason is None:
        energies.append(atoms.get_potential_energy(force_consistent=force_consistent))
        positions.append(atoms.positions)
        uncertainties.append(atoms.get_calculator().results['uncertainty'])
        indexes_minima = find_peaks(-np.array(energies))[0]

        dyn.run(1)  # Run MD step.
        # A. Stop MD simulation if max. uncertainty is crossed over.
        if uncertainties[-1] > maxstep:
            stop_reason = 'max_uncertainty_reached'
        # B. Stop MD simulation if 'mdmin' number of minima are found.
        elif len(indexes_minima) >= mdmin:
            stop_reason = 'mdmin_found'
        # C. Stop if max. time has been reached.
        elif current_time >= maxtime:
            stop_reason = 'max_time_reached'
        # D. Stop if energy threshold has been crossed over (e_threshold).
        elif energies[-1] >= initial_energy + energy_threshold:
            stop_reason = 'max_energy_reached'
        else:
            current_time += timestep  # Update the time of the MD simulation.

    indexes_maxima = find_peaks(np.array(energies))[0]
    results = {'stop_reason': stop_reason, 'indexes_minima': indexes_minima,
               'indexes_maxima': indexes_maxima, 'energies': energies,
               'uncertainties': uncertainties, 'positions': positions}
    return results


@parallel_function
def get_fmax(atoms):
    """
    Returns fmax for a given atoms structure.
    """
    forces = atoms.get_forces()
    return np.sqrt((forces**2).sum(axis=1).max())


@parallel_function
def dump_trajectory(atoms, filename, restart):
    """
    Saves a trajectory file containing the atoms observations.

    Parameters
    ----------
    atoms: object
        Atoms object to be appended to previous observations.
    filename: string
        Name of the trajectory file to save the observations.
    restart: boolean
        Append mode (true or false).
     """

    if restart is True:
        try:
            prev_atoms = io.read(filename, ':')  # Actively searching.
            if atoms not in prev_atoms:  # Avoid duplicates.
                # Update observations.
                new_atoms = prev_atoms + [atoms]
                io.write(filename=filename, images=new_atoms)
        except Exception:
            io.write(filename=filename, images=atoms, append=False)
    if restart is False:
        io.write(filename=filename, images=atoms, append=False)

