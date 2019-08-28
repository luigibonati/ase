import numpy as np
import copy
import os
from ase.optimize.activelearning.gp.calculator import GPCalculator
from ase.optimize.activelearning.gp.calculator import ConstantPrior
from ase import io, units
from ase.parallel import parprint, parallel_function
from ase.optimize.activelearning.acquisition import acquisition
from ase.md import VelocityVerlet
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.optimize.minimahopping import ComparePositions
from ase.optimize.activelearning.aidmin import AIDMin
from ase.optimize import *
from ase.optimize.activelearning.io import get_fmax, dump_observation


class AIDHopping:

    def __init__(self, atoms, calculator, model_calculator=None,
                 force_consistent=None, max_train_data=50,
                 max_train_data_strategy='nearest_observations',
                 trajectory='AID.traj', T0=500., beta1=1.01,
                 beta2=0.98, beta3=0.75, beta4=0.02, energy_threshold=2.5,
                 geometry_threshold=1., maxstep=.5, timestep=1.0,
                 maxtime=1000., maxoptsteps=500):
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
        self.temperature = T0
        self.energy_threshold = energy_threshold
        self.timestep = timestep
        self.maxtime = maxtime
        self.maxstep = maxstep
        self.maxoptsteps = maxoptsteps

        # Hopping parameters.
        self.beta1 = beta1  # Increase temperature.
        self.beta2 = beta2  # Decrease temperature.
        self.beta3 = beta3  # Decrease temperature after finding a new minimum.
        self.beta4 = beta4  # Increase energy threshold (in eV).
        self.geometry_threshold = geometry_threshold

        # Model parameters.
        self.model_calculator = model_calculator
        # Default GP Calculator parameters if not specified by the user.
        if model_calculator is None:
            self.model_calculator = GPCalculator(
                            train_images=[],
                            scale=1., weight=2.,
                            update_prior_strategy='maximum',
                            max_train_data=max_train_data,
                            max_train_data_strategy=max_train_data_strategy,
                            wrap_positions=False)
        self.model_calculator.calculate_uncertainty = True

        # Active Learning setup (single-point calculations).
        self.function_calls = 0
        self.force_calls = 0
        self.ase_calc = calculator
        self.atoms = atoms
        self.fmax = None

        self.constraints = self.atoms.constraints
        self.fc = force_consistent
        self.trajectory = trajectory
        trajectory_main = self.trajectory.split('.')[0]
        self.trajectory_observations = trajectory_main + '_observations.traj'
        self.trajectory_candidates = trajectory_main + '_candidates.traj'
        self.trajectory_minima = trajectory_main + '_found_minima.traj'

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
        self.fmax = fmax

        # Initialize, check whether previous minima have been encountered.
        if os.path.exists(self.trajectory_observations):
            observations = io.read(self.trajectory_observations, ':')
            for observation in observations:
                if get_fmax(observation) <= self.fmax:
                    restart = True
                    if not os.path.exists(self.trajectory_minima):
                        restart = False
                    dump_observation(atoms=observation,
                                     filename=self.trajectory_minima,
                                     restart=restart)

        if not os.path.exists(self.trajectory_minima):
            self.atoms.set_calculator(self.ase_calc)
            opt = AIDMin(self.atoms, trajectory=self.trajectory,
                         use_previous_observations=True)
            opt.run(fmax=self.fmax)
            dump_observation(opt.atoms, filename=self.trajectory_minima,
                             restart=False)

        self.temperature = self.T0
        while self.function_calls <= steps:

            train_images = io.read(self.trajectory_observations, ':')
            gp_calc = copy.deepcopy(self.model_calculator)
            gp_calc.update_train_data(train_images=train_images)
            prev_minima = io.read(self.trajectory_minima, ':')

            # Perform optimizations starting from each minimum found.
            # np.random.seed(int(len(prev_minima) * self.temperature))

            candidates = []
            while len(candidates) < 1:
                for index_minimum in range(0, len(prev_minima)):
                    parprint('Starting from minimum:', index_minimum)
                    md_guess = copy.deepcopy(prev_minima[index_minimum])
                    md_guess.set_calculator(gp_calc)
                    md_guess.get_potential_energy(force_consistent=self.fc)
                    stop_reason = mdsim(atoms=md_guess, maxstep=self.maxstep,
                                        train_images=train_images,
                                        temperature=self.temperature,
                                        timestep=self.timestep,
                                        maxtime=self.maxtime,
                                        energy_threshold=self.energy_threshold,
                                        trajectory='md_simulation.traj',
                                        force_consistent=self.fc)

                    parprint('MD stopping reason:', stop_reason)

                    if stop_reason == 'max_time_reached':
                        parprint('Increase temp. due to max. time reached.')
                        self.temperature *= self.beta1  # Increase temperature.

                    if stop_reason == 'max_energy_reached':
                        parprint('Decrease temp. due to max. energy reached.')
                        self.temperature = self.T0  # Decrease temperature.
                        parprint('Increase energy threshold to explore '
                                 'higher energy regions.')
                        self.energy_threshold += self.beta4  # Increase energy.
                        parprint('Current energy threshold is: ',
                                 self.energy_threshold)

                    if stop_reason == 'max_uncertainty_reached':
                        candidates += [copy.deepcopy(md_guess)]
                        self.temperature *= self.beta2
                        # self.temperature = self.T0

                    if stop_reason == 'mdmin_found':
                        opt_atoms = io.read(self.trajectory_minima, -1)
                        model_min = GPCalculator(
                                        train_images=[],
                                        scale=0.3, weight=2.,
                                        update_prior_strategy='maximum',
                                        max_train_data=50,
                                        wrap_positions=False
                                        )
                        model_min.update_train_data(
                                        train_images=train_images,
                                        test_images=[opt_atoms]
                                        )
                        opt_atoms.positions = md_guess.positions
                        gp_calc_min = copy.deepcopy(model_min)
                        gp_calc_min.update_train_data(
                                                     train_images=train_images,
                                                     test_images=[opt_atoms]
                                                     )
                        opt_atoms.set_calculator(gp_calc_min)
                        opt_min = QuasiNewton(opt_atoms, logfile=None)
                        opt_min.run(fmax=self.fmax*0.01)

                        # Check whether duplicate.
                        unique_candidate = _check_unique_minima_found(
                                                          self,
                                                          atoms=opt_min.atoms
                                                          )
                        if unique_candidate is True:
                            candidates += [opt_min.atoms]
                        else:
                            parprint('Re-found minima...increasing temp.')
                            self.temperature *= self.beta1

                        # Obey threshold of minimum temperature.
                        if self.temperature < self.T0:
                            self.temperature = self.T0

            sorted_candidates = acquisition(train_images=train_images,
                                            candidates=candidates,
                                            mode='ucb',
                                            objective='max'
                                            )

            best_candidate = sorted_candidates.pop(0)

            # Perform Minimization.
            self.atoms.positions = best_candidate.positions
            self.atoms.set_calculator(self.ase_calc)
            self.atoms.get_potential_energy(force_consistent=self.fc)
            self.atoms.get_forces()

            min_prior = self.atoms.get_potential_energy(
                                                      force_consistent=self.fc
                                                      )
            model_min_greedy = GPCalculator(train_images=[],
                                            scale=0.3, weight=2.,
                                            prior=ConstantPrior(min_prior),
                                            update_prior_strategy=None,
                                            max_train_data=5,
                                            wrap_positions=False
                                            )
            while True:
                opt_min_greedy = AIDMin(
                              self.atoms,
                              use_previous_observations=True,
                              trajectory=self.trajectory,
                              model_calculator=copy.deepcopy(model_min_greedy)
                              )
                opt_min_greedy.run(fmax=self.fmax, steps=0)

                opt_unique = _check_unique_minima_found(
                                                    self,
                                                    atoms=opt_min_greedy.atoms
                                                    )
                prev_opt = io.read(self.trajectory_observations, '-1')

                if not opt_unique:
                    parprint('Previously found structure...do not evaluate.')
                    self.temperature *= self.beta1
                    break

                if opt_unique:
                    self.atoms.positions = opt_min_greedy.atoms.positions
                    self.atoms.set_calculator(self.ase_calc)
                    self.atoms.get_potential_energy(force_consistent=self.fc)
                    if get_fmax(prev_opt) <= self.fmax:
                        self.temperature *= self.beta3
                        parprint('Re-starting temperature...')
                        dump_observation(atoms=prev_opt,
                                         filename=self.trajectory_minima,
                                         restart=True)
                        break

            file_function_calls = io.read(self.trajectory_observations, ':')
            self.function_calls = len(file_function_calls)
            prev_minima = io.read(self.trajectory_minima, ':')
            parprint('-' * 78)
            parprint('Function calls:', self.function_calls)
            parprint('Minima found:', len(prev_minima))
            parprint('-' * 78)


@parallel_function
def mdsim(atoms, temperature,  train_images, timestep=1.0, maxstep=0.5,
          maxtime=2000., energy_threshold=2.0, force_consistent=None,
          trajectory='md_simulation.traj'):
    """Performs a molecular dynamics simulation until any of the following
    conditions is achieved: (A) 'mdmin' number of minima are found or  or (B)
    'max_time' (in fs) has been reached or (C) energy threshold has been
    crossed over ('energy_threshold') or (D) maximum uncertainty reached
    (determined by 'max_step')."""

    parprint('Current temperature for MD:', temperature)
    initial_energy = atoms.get_potential_energy(
                                        force_consistent=force_consistent
                                        )
    current_time = 0.0  # Initial time.
    energies, uncertainties, positions, indexes_minima = [], [], [], []
    stop_reason = None

    MaxwellBoltzmannDistribution(atoms, temp=temperature * units.kB,
                                 force_temp=True)
    dyn = VelocityVerlet(atoms, timestep=timestep * units.fs,
                         trajectory=trajectory)
    step = 0
    while stop_reason is None:
        if step % 100 == 0:
            atoms.get_calculator().update_train_data(train_images=train_images,
                                                     test_images=[atoms])
        energies.append(
                  atoms.get_potential_energy(force_consistent=force_consistent)
                  )
        uncertainties.append(atoms.get_calculator().results['uncertainty'])
        md_value = np.array(energies)
        indexes_minima = passed_minimum(nup=5, ndown=5, energies=md_value)
        dyn.run(1)  # Run MD step.

        # A. Stop MD simulation if 'mdmin' number of minima are found.
        if indexes_minima is not None:
            stop_reason = 'mdmin_found'
        # B. Stop if max. time has been reached.
        elif current_time >= maxtime:
            stop_reason = 'max_time_reached'
        # C. Stop if energy threshold has been crossed over (e_threshold).
        elif energies[-1] >= initial_energy + energy_threshold:
            stop_reason = 'max_energy_reached'
        # D. Max uncertainty reached.
        elif uncertainties[-1] > maxstep:
            stop_reason = 'max_uncertainty_reached'

        current_time += timestep  # Update the time of the MD simulation.
        step += 1
    return stop_reason


@parallel_function
def _check_unique_minima_found(self, atoms):
    prev_minima = io.read(self.trajectory_minima, ':')
    unique_candidate = True
    for minimum in prev_minima:
        compare = ComparePositions(translate=True)
        dmax = compare(atoms1=atoms, atoms2=minimum)
        if dmax <= self.geometry_threshold:
            unique_candidate = False
    return unique_candidate


@parallel_function
def passed_minimum(nup=2, ndown=2, energies=None):
    if len(energies) < (nup + ndown + 1):
        return None
    status = True
    index = -1
    for i_up in range(nup):
        if energies[index] < energies[index - 1]:
            status = False
        index -= 1
    for i_down in range(ndown):
        if energies[index] > energies[index - 1]:
            status = False
        index -= 1
    if status:
        return (-nup - 1), energies[-nup - 1]
