import numpy as np
import time
import copy
from ase import io
from ase.calculators.gp.calculator import GPCalculator
from ase.optimize.lbfgs import LBFGS
from ase.parallel import parprint, parallel_function


class LGPMin:

    def __init__(self, atoms, model_calculator=None, force_consistent=None,
                 max_train_data=20,
                 max_train_data_strategy='last_observations'):
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

        """

        # Default GP Calculator parameters if not specified by the user.
        self.model_calculator = model_calculator
        if model_calculator is None:
            self.model_calculator = GPCalculator(
                               train_images=[],
                               max_train_data_strategy=max_train_data_strategy,
                               max_train_data=max_train_data)
        # GPMin does not uses uncertainty (switch off for faster predictions).
        self.model_calculator.calculate_uncertainty = False

        # Active Learning setup (Single-point calculations).
        self.function_calls = 0
        self.force_calls = 0
        self.ase_calc = atoms.get_calculator()
        self.atoms = atoms

        self.constraints = self.atoms.constraints
        self.force_consistent = force_consistent

    def run(self, fmax=0.05, ml_steps=1000, trajectory='LGPMin.traj',
            restart=False):

        """
        Executing run will start the optimization process.

        Parameters
        ----------
        fmax : float
            Convergence criteria (in eV/Angstrom).

        ml_steps: int
            Maximum number of steps for the optimization (using LBFGS) on
            the GP predicted potential energy surface.

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

        Returns
        -------
        Optimized structure. The optimization process can be followed in
        *trajectory_observations.traj*.

        """
        trajectory_main = trajectory.split('.')[0]
        trajectory_observations = trajectory_main + '_observations.traj'
        trajectory_candidates = trajectory_main + '_candidates.traj'

        # Start by saving the initial configurations.
        # If restart is True it will read previous observations from
        # *trajectory_observations.traj* if the file is found in the working
        # directory. If restart is False it will overwrite any previous
        # *trajectory_observations.traj* and will start the optimization from
        # scratch.

        self.atoms.get_potential_energy(force_consistent=self.force_consistent)
        self.atoms.get_forces()
        dump_observation(atoms=self.atoms,
                         filename=trajectory_observations, restart=restart)
        train_images = io.read(trajectory_observations, ':')

        while not fmax > get_fmax(train_images[-1]):

            # 1. Collect observations.
            # This serves to restart from a previous (and/or parallel) runs.
            start = time.time()
            train_images = io.read(trajectory_observations, ':')
            end = time.time()
            parprint('Time reading and writing atoms images to build a model:',
                     end-start)

            # 2. Update GP calculator.
            gp_calc = copy.deepcopy(self.model_calculator)
            gp_calc.update_train_data(train_images=train_images)
            self.atoms.set_calculator(gp_calc)

            # 3. Optimize the structure in the predicted PES.
            ml_opt = LBFGS(self.atoms, trajectory=trajectory_candidates)
            ml_opt.run(fmax=(fmax * 0.5), steps=ml_steps)

            # 4. Evaluate the target function and save it in *observations*.
            parprint('Performing evaluation on the real landscape...')
            self.atoms.set_calculator(self.ase_calc)
            self.atoms.get_potential_energy(force_consistent=self.force_consistent)
            self.atoms.get_forces()
            dump_observation(atoms=self.atoms,
                             filename=trajectory_observations, restart=True)
            self.function_calls += 1
            self.force_calls += 1
            parprint('Single-point calculation finished.')

            # 6. Print output.
            msg = "--------------------------------------------------------"
            parprint(msg)
            parprint('Step:', self.function_calls)
            parprint('Time:', time.strftime("%m/%d/%Y, %H:%M:%S",
                                            time.localtime()))
            parprint('Energy', self.atoms.get_potential_energy(self.force_consistent))
            parprint("fmax:", get_fmax(train_images[-1]))
            msg = "--------------------------------------------------------\n"
            parprint(msg)
        print_cite_min()


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
def print_cite_min():
    msg = "\n" + "-" * 79 + "\n"
    msg += "You are using LGPMin. Please cite: \n"
    msg += "[1] E. Garijo del Rio, J. J. Mortensen and K. W. Jacobsen. "
    msg += "arXiv:1808.08588. https://arxiv.org/abs/1808.08588. \n"
    msg += "[1] M. H. Hansen, J. A. Garrido Torres, P. C. Jennings, "
    msg += "J. R. Boes, O. G. Mamun and T. Bligaard. arXiv:1904.00904. "
    msg += "https://arxiv.org/abs/1904.00904 \n"
    msg += "-" * 79 + "\n"
    parprint(msg)
