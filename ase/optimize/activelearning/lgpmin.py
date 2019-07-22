import numpy as np
import time
import copy
from ase import io
from ase.calculators.gp.calculator import GPCalculator
from ase.parallel import parprint, parallel_function
from scipy.spatial.distance import euclidean
from ase.optimize import *


class LGPMin:

    def __init__(self, atoms, model_calculator=None, force_consistent=None,
                 max_train_data=5, optimizer=QuasiNewton,
                 max_train_data_strategy='nearest_observations',
                 geometry_threshold=0.001, trajectory='LGPMin.traj',
                 restart=False):
        """
        Optimize atomic structure using a surrogate machine learning
        model [1,2]. Potential energies and forces information are used to
        build a predicted PES via Gaussian Process (GP) regression.
        [1] E. Garijo del Rio, J. J. Mortensen and K. W. Jacobsen.
        arXiv:1808.08588.
        [2] M. H. Hansen, J. A. Garrido Torres, P. C. Jennings, Z. Wang,
        J. R. Boes, O. G. Mamun and T. Bligaard. arXiv:1904.00904.

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

        restart: boolean
            A *trajectory_observations.traj* file is automatically generated
            which contains the observations collected by the surrogate. If
            *restart* is True and a *trajectory_observations.traj* file is
            found in the working directory it will be used to continue the
            optimization from previous run(s). In order to start the
            optimization from scratch *restart* should be set to False or
            alternatively the *trajectory_observations.traj* file must be
            deleted.

        """
        self.model_calculator = model_calculator
        # Default GP Calculator parameters if not specified by the user.
        if model_calculator is None:
            self.model_calculator = GPCalculator(
                               train_images=[], scale=0.3, weight=2.0,
                               max_train_data_strategy=max_train_data_strategy,
                               max_train_data=max_train_data)

        # GPMin does not use uncertainty (switched off for faster predictions).
        self.model_calculator.calculate_uncertainty = False

        # Active Learning setup (single-point calculations).
        self.function_calls = 0
        self.force_calls = 0
        self.step = 0

        self.atoms = atoms
        self.ase_calc = atoms.get_calculator()
        self.optimizer = optimizer

        self.fc = force_consistent
        self.trajectory = trajectory
        self.restart = restart
        self.geometry_threshold = geometry_threshold

        trajectory_main = self.trajectory.split('.')[0]
        self.trajectory_observations = trajectory_main + '_observations.traj'

        self.atoms.get_potential_energy()
        self.atoms.get_forces()

        dump_observation(atoms=self.atoms,
                         filename=self.trajectory_observations,
                         restart=self.restart)

    def run(self, fmax=0.05, ml_steps=1000, steps=200, logfile=False):

        """
        Executing run will start the optimization process.

        Parameters
        ----------
        fmax : float
            Convergence criteria (in eV/Angstrom).

        ml_steps: int
            Maximum number of steps for the optimization (using LBFGS) on
            the GP predicted potential energy surface.

        steps: int
            Maximum number of steps for the surrogate.

        Returns
        -------
        Optimized structure. The optimization process can be followed in
        *trajectory_observations.traj*.

        """
        self.fmax = fmax
        self.steps = steps


        # Always start from 'atoms' positions.
        starting_atoms = io.read(self.trajectory_observations, -1)
        starting_atoms.positions = copy.deepcopy(self.atoms.positions)

        while not self.fmax >= get_fmax(self.atoms):

            # 1. Collect observations.
            # This serves to restart from a previous (and/or parallel) runs.
            train_images = io.read(self.trajectory_observations, ':')

            # 2. Update GP calculator.
            # Probed positions are used for low-memory.

            ml_converged = False

            surrogate_positions = self.atoms.positions
            probed_atoms = [copy.deepcopy(starting_atoms)]
            probed_atoms[0].positions = copy.deepcopy(self.atoms.positions)

            while not ml_converged:
                gp_calc = copy.deepcopy(self.model_calculator)
                gp_calc.update_train_data(train_images=train_images,
                                          test_images=probed_atoms)
                self.atoms.set_calculator(gp_calc)

                # 3. Optimize the structure in the predicted PES.
                ml_opt = self.optimizer(self.atoms,
                                        logfile=None, trajectory=None)
                ml_opt.run(fmax=(fmax * 0.01), steps=ml_steps)
                surrogate_positions = self.atoms.positions

                if len(probed_atoms) >= 2:
                    l1_probed_pos = probed_atoms[-1].positions.reshape(-1)
                    l2_probed_pos = probed_atoms[-2].positions.reshape(-1)
                    dl1l2 = euclidean(l1_probed_pos, l2_probed_pos)
                    if dl1l2 <= self.geometry_threshold:
                        ml_converged = True

                probed = copy.deepcopy(probed_atoms[0])
                probed.positions = copy.deepcopy(self.atoms.positions)
                probed_atoms += [probed]

            # Update step (this allows to stop algorithm before evaluating).
            if self.step >= self.steps:
                break

            # 4. Evaluate the target function and save it in *observations*.
            # Update the new positions.
            self.atoms.positions = surrogate_positions
            self.atoms.set_calculator(self.ase_calc)
            self.atoms.get_potential_energy(force_consistent=self.fc)
            self.atoms.get_forces()

            dump_observation(atoms=self.atoms,
                             filename=self.trajectory_observations,
                             restart=True)

            self.function_calls = len(train_images) + 1
            self.force_calls = self.function_calls
            self.step += 1

            # 5. Print output.
            if logfile is True:
                _log(self)


@parallel_function
def get_fmax(atoms):
    """
    Returns fmax for a given atoms structure.
    """
    forces = atoms.get_forces()
    return np.sqrt((forces**2).sum(axis=1).max())


@parallel_function
def dump_observation(atoms, filename, restart):
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


@parallel_function
def _log(self):
    msg = "-" * 26
    parprint(msg)
    parprint('Step:', self.step)
    parprint('Function calls:', self.function_calls)
    parprint('Time:', time.strftime("%m/%d/%Y, %H:%M:%S", time.localtime()))
    parprint('Energy:', self.atoms.get_potential_energy(self.fc))
    parprint("fmax:", get_fmax(self.atoms))
    msg = "-" * 26 + "\n"
    parprint(msg)
