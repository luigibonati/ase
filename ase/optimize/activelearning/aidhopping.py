import numpy as np
import copy
import os
from ase.optimize.activelearning.gp.calculator import GPCalculator
from ase import io, units
from ase.parallel import parprint, parallel_function
from ase.optimize.activelearning.acquisition import acquisition
from ase.md import VelocityVerlet
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.optimize.minimahopping import ComparePositions
from ase.optimize.activelearning.aidmin import AIDMin
from ase.optimize.activelearning.io import get_fmax, dump_observation
from ase.optimize.activelearning.gp.prior import ConstantPrior


class AIDHopping:

    def __init__(self, atoms, calculator, model_calculator=None,
                 force_consistent=None, max_train_data=500,
                 max_train_data_strategy='nearest_observations',
                 trajectory='AID.traj', T0=500., geometry_threshold=1.,
                 maxstep=0.75):
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
        self.maxstep = maxstep
        self.initial_maxstep = maxstep
        self.max_train_data = max_train_data
        self.max_train_data_strategy = max_train_data_strategy

        # Hopping parameters.
        self.geometry_threshold = geometry_threshold

        # Model parameters.
        self.model_calculator = model_calculator
        # Default GP Calculator parameters if not specified by the user.

        # Active Learning setup (single-point calculations).
        self.function_calls = 0
        self.force_calls = 0
        self.ase_calc = calculator
        self.atoms = atoms
        self.fmax = None
        self.step = 0

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

        # Start with random seed
        prev_minima = io.read(self.trajectory_minima, ':')

        random_direction = int(len(prev_minima) * self.temperature)

        while self.step <= steps:

            np.random.seed(random_direction)
            parprint('Random direction seed:', random_direction)

            train_images = io.read(self.trajectory_observations, ':')

            candidates = []

            md_guess = copy.deepcopy(prev_minima[-1])
            e_prior_minimum = md_guess.get_potential_energy()

            prior = ConstantPrior(constant=e_prior_minimum)
            gp_calc = GPCalculator(
                    train_images=[],
                    scale=0.5, weight=1., update_hyperparams=False,
                    fit_weight=None,
                    update_prior_strategy=None, prior=prior,
                    max_train_data=self.max_train_data,
                    max_train_data_strategy=self.max_train_data_strategy,
                    wrap_positions=False
                    )

            gp_calc.update_train_data(train_images=train_images,
                                      test_images=[copy.deepcopy(
                                                   md_guess)])
            md_guess.set_calculator(gp_calc)
            md_guess.get_potential_energy(force_consistent=self.fc)

            md_atoms = md_simulation(atoms=md_guess, maxstep=self.maxstep,
                                     temperature=self.temperature)
            candidates += [md_atoms]

            sorted_candidates = acquisition(train_images=train_images,
                                            candidates=candidates,
                                            mode='fmax',
                                            objective='max'
                                            )

            best_candidate = sorted_candidates.pop(0)

            # Perform Minimization.
            self.atoms.positions = best_candidate.positions
            self.atoms.set_calculator(self.ase_calc)
            self.atoms.get_potential_energy(force_consistent=self.fc)

            self.atoms.get_forces()

            model_min_greedy = GPCalculator(train_images=[],
                                            scale=0.35, weight=2.,
                                            update_prior_strategy='fit',
                                            max_train_data=5,
                                            fit_weight=None,
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

                if not opt_unique:
                    parprint('Previously found structure...do not evaluate.')
                    break

                if opt_unique:
                    self.atoms.positions = opt_min_greedy.atoms.positions
                    self.atoms.set_calculator(self.ase_calc)
                    self.atoms.get_potential_energy(force_consistent=self.fc)
                    if get_fmax(self.atoms) <= self.fmax:
                        dump_observation(atoms=self.atoms,
                                         filename=self.trajectory_minima,
                                         restart=True)
                        break


                print('energy minima:', prev_minima[0].get_potential_energy())
                print('energy now:', self.atoms.get_potential_energy())
                print('delta', self.atoms.get_potential_energy() - prev_minima[0].get_potential_energy())
                if self.atoms.get_potential_energy() - prev_minima[
                                        0].get_potential_energy() > 1.5:
                    random_direction += 1
                    break

            file_function_calls = io.read(self.trajectory_observations, ':')
            self.function_calls = len(file_function_calls)
            prev_minima = io.read(self.trajectory_minima, ':')
            parprint('-' * 78)
            parprint('Step:', self.step)
            parprint('Function calls:', self.function_calls)
            parprint('Minima found:', len(prev_minima))
            parprint('Energy:', self.atoms.get_potential_energy(
                                            force_consistent=self.fc))

            parprint('-' * 78)
            self.step += 1


@parallel_function
def md_simulation(atoms, temperature,  maxstep):
    """Performs a molecular dynamics simulation until maximum uncertainty
    reached (determined by 'maxstep')."""

    parprint('Current temperature for MD:', temperature)
    MaxwellBoltzmannDistribution(atoms, temp=temperature * units.kB,
                                 force_temp=True)
    dyn = VelocityVerlet(atoms, timestep=5. * units.fs,
                         trajectory=None)
    atoms_unc = atoms.get_calculator().results['uncertainty']
    while atoms_unc < maxstep:
        atoms_unc = atoms.get_calculator().results['uncertainty']
        dyn.run(1)  # Run MD step.
    return atoms


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
