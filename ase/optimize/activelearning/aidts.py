import numpy as np
import time
import copy
from ase import io
from ase.optimize.activelearning.gp.calculator import GPCalculator
from ase.parallel import parprint
from ase.dimer import DimerControl, MinModeAtoms, MinModeTranslate
from ase.optimize.activelearning.io import get_fmax, dump_observation


class AIDTS:

    def __init__(self, atoms, atoms_vector, vector_length=0.01,
                 model_calculator=None, force_consistent=None,
                 trajectory='AIDTS.traj', max_train_data=50,
                 max_train_data_strategy='nearest_observations',
                 restart=False):
        """
        Artificial Intelligence-Driven dimer (AID-TS) algorithm.
        Dimer optimization of an atomic structure using a surrogate machine
        learning model. Potential energies and forces information are used to
        build a model potential energy surface (PES). A dimer is launched
        from an initial 'atoms' structure toward the direction of the
        'atoms_vector' structure with a magnitude of 'vector_length'.
        The code automatically recognizes the atoms that are not involved
        in the displacement and their constraints.

        Parameters
        --------------
        atoms: Atoms object
            The Atoms object to relax.

        atoms_vector: Atoms object.
            Dummy Atoms object with the structure to use for the saddle-point
            search direction. The coordinates of this structure serve to
            build the vector used for the dimer optimization. Therefore,
            the 'atoms' structure will be "pushed" uphill in the PES along the
            direction of the 'atoms_vector'.

        model_calculator: Model object.
            Model calculator to be used for predicting the PES. The default
            is None which uses a GP model with the Squared Exponential
            Kernel and other default parameters. See
            *ase.calculator.gp.calculator* GPModel for default GP parameters.

        force_consistent: boolean or None
            Use force-consistent energy calls (as opposed to the energy
            extrapolated to 0 K). By default (force_consistent=None) uses
            force-consistent energies if available in the calculator, but
            falls back on force_consistent=False if not.

                trajectory: string
            Filename to store the predicted optimization.
                Additional information:
                - Uncertainty: The energy uncertainty in each image can be
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
                               max_train_data=max_train_data,
                               update_prior_strategy='maximum')

        # AID-TS doesn't use uncertainty (switched off for faster predictions).
        self.model_calculator.calculate_uncertainty = False

        # Active Learning setup (single-point calculations).
        self.function_calls = 0
        self.force_calls = 0
        self.step = 0

        self.atoms = atoms
        self.atoms_vector = atoms_vector

        # Calculate displacement vector and mask automatically.
        displacement_vector = []
        mask_atoms = []

        for atom in range(0, len(atoms)):
            vect_atom = (atoms_vector[atom].position - atoms[atom].position)
            displacement_vector += [vect_atom.tolist()]
            if np.array_equal(np.array(vect_atom), np.array([0, 0, 0])):
                mask_atoms += [0]
            else:
                mask_atoms += [1]

        max_vector = np.max(np.abs(displacement_vector))
        normalized_vector = (np.array(displacement_vector) / max_vector)
        normalized_vector *= vector_length
        self.displacement_vector = normalized_vector

        self.mask_atoms = mask_atoms
        self.vector_length = vector_length

        # Optimization settings.
        self.ase_calc = atoms.get_calculator()
        self.fc = force_consistent
        self.trajectory = trajectory
        self.restart = restart

        trajectory_main = self.trajectory.split('.')[0]
        self.trajectory_observations = trajectory_main + '_observations.traj'

        # First observation calculation.
        self.atoms.get_potential_energy()
        self.atoms.get_forces()

        dump_observation(atoms=self.atoms, method='dimer',
                         filename=self.trajectory_observations,
                         restart=self.restart)

    def run(self, fmax=0.05, steps=200, logfile=True):

        """
        Executing run will start the optimization process.

        Parameters
        ----------
        fmax : float
            Convergence criteria (in eV/Angstrom).

        steps: int
            Maximum number of steps for the surrogate.

        Returns
        -------
        Optimized structure. The optimization process can be followed in
        *'trajectory'_observations.traj*.

        """
        self.fmax = fmax
        self.steps = steps

        initial_atoms = copy.deepcopy(self.atoms)

        # Probed atoms are used to know the path followed for low memory.
        probed_atoms = [io.read(self.trajectory_observations, '-1')]

        while True:

            # 1. Collect observations.
            # This serves to restart from a previous (and/or parallel) runs.
            train_images = io.read(self.trajectory_observations, ':')

            # 2. Update model calculator.

            # Start from initial structure positions.
            self.atoms.positions = initial_atoms.positions

            gp_calc = copy.deepcopy(self.model_calculator)
            gp_calc.update_train_data(train_images=train_images,
                                      test_images=probed_atoms)
            self.atoms.set_calculator(gp_calc)

            # 3. Optimize dimer in the predicted PES.
            d_control = DimerControl(initial_eigenmode_method='displacement',
                                     displacement_method='vector',
                                     logfile=None, use_central_forces=False,
                                     extrapolate_forces=False,
                                     displacement_radius=self.vector_length,
                                     mask=self.mask_atoms)
            d_atoms = MinModeAtoms(self.atoms, d_control)
            d_atoms.displace(displacement_vector=self.displacement_vector)
            dim_rlx = MinModeTranslate(d_atoms, trajectory=None)
            dim_rlx.run(fmax=fmax*0.1)

            surrogate_positions = self.atoms.positions

            # Probed atoms serve to track dimer structures for low memory.

            surrogate_atoms = copy.deepcopy(probed_atoms[0])
            surrogate_atoms.positions = surrogate_positions
            probed_atoms += [surrogate_atoms]

            # Update step (this allows to stop algorithm before evaluating).
            if self.step >= self.steps:
                break

            # 4. Evaluate the target function and save it in *observations*.
            # Update the new positions.
            self.atoms.positions = surrogate_positions
            self.atoms.set_calculator(self.ase_calc)
            self.atoms.get_potential_energy(force_consistent=self.fc)
            self.atoms.get_forces()

            dump_observation(atoms=self.atoms, method='dimer',
                             filename=self.trajectory_observations,
                             restart=True)

            self.function_calls = len(train_images) + 1
            self.force_calls = self.function_calls
            self.step += 1

            # 5. Print output.
            if logfile is True:
                parprint("-" * 26)
                parprint('Step:', self.step)
                parprint('Function calls:', self.function_calls)
                parprint('Time:', time.strftime("%m/%d/%Y, %H:%M:%S", time.localtime()))
                parprint('Energy:', self.atoms.get_potential_energy(self.fc))
                parprint("fmax:", get_fmax(self.atoms))
                parprint("-" * 26 + "\n")

            if get_fmax(self.atoms) <= self.fmax:
                parprint('Converged.')
                break
