from ase import io
from ase.optimize.activelearning.gp.calculator import GPCalculator
from ase.parallel import parprint, parallel_function
from ase.optimize.activelearning.aidmin import AIDMin
from ase.optimize.activelearning.aidts import AIDTS
from ase.optimize.activelearning.io import TrainingSet
import os


class AIDMEP:

    def __init__(self, images, calculator, model_calculator=None,
                 max_train_data=5, force_consistent=None,
                 max_train_data_strategy='nearest_observations',
                 trajectory='AID.traj',
                 use_previous_observations=True,
                 trainingset='AID_observations.traj'):

        """
        Artificial Intelligence-Driven Minimum Energy Pathway refine method.
        Refine a set of images (usually from an optimized AIDNEB
        calculation) using the previously collected observations [1-3].
        Potential energies and forces at a given position are
        supplied to the model calculator to build a modelled PES in an
        active-learning fashion. This surrogate relies on the AIDMin [2] and
        AIDNEB [3] algorithms and takes a set of images to perform
        minimizations and dimer optimization in the predicted PES to obtain
        minima and transition-states.

        [1] J. A. Garrido Torres, E. Garijo del Rio, V. Streibel,
        T. S. Choski, J. J. Mortensen, A. Urban, M. Bajdich,
        F. Abild-Pedersen, K. W. Jacobsen, and T. Bligaard (submitted).
        [2] E. Garijo del Rio, J. J. Mortensen and K. W. Jacobsen. Phys.
        Rev. B 100, 104103.
        https://journals.aps.org/prb/abstract/10.1103/PhysRevB.100.104103
        [3] J. A. Garrido Torres, M. H. Hansen, P. C. Jennings, J. R. Boes
        and T. Bligaard. Phys. Rev. Lett. 122, 156001.
        https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.122.156001


        Parameters
        --------------
        images: Trajectory file (in ASE format) or Atoms object.
            Set of images containing

        final: Trajectory file (in ASE format) or Atoms object.
            Final end-point of the NEB path.

        model_calculator: Model object.
            Model calculator to be used for predicting the potential energy
            surface. The default is None which uses a GP model with the Squared
            Exponential Kernel and other default parameters. See
            *ase.optimize.activelearning.gp.calculator* GPModel for default GP
            parameters.

        calculator: ASE calculator Object.
            ASE calculator.
            See https://wiki.fysik.dtu.dk/ase/ase/calculators/calculators.html

        force_consistent: boolean or None
            Use force-consistent energy calls (as opposed to the energy
            extrapolated to 0 K). By default (force_consistent=None) uses
            force-consistent energies if available in the calculator, but
            falls back on force_consistent=False if not.

        trajectory: string
            Filename to store the predicted NEB paths.
                Additional information:
                - Energy uncertain: The energy uncertainty in each image
                position can be accessed in image.info['uncertainty'].

        use_previous_observations: boolean
            If False. The optimization starts from scratch.
            A *trajectory_observations.traj* file is automatically generated
            in each step of the optimization, which contains the
            observations collected by the surrogate. If
            (a) *use_previous_observations* is True and (b) a previous
            *trajectory_observations.traj* file is found in the working
            directory: the algorithm will be use the previous observations
            to train the model with all the information collected in
            *trajectory_observations.traj*.

        max_train_data: int
            Number of observations that will effectively be included in the
            model. See also *max_data_strategy*.

        max_train_data_strategy: string
            Strategy to decide the observations that will be included in the
            model.

            options:
                'last_observations': selects the last observations collected by
                the surrogate.
                'lowest_energy': selects the lowest energy observations
                collected by the surrogate.
                'nearest_observations': selects the observations which
                positions are nearest to the positions of the Atoms to test.

            For instance, if *max_train_data* is set to 50 and
            *max_train_data_strategy* to 'lowest energy', the surrogate model
            will be built in each iteration with the 50 lowest energy
            observations collected so far.

        """

        if isinstance(images, list):
            io.write('initial_path.traj', images)
            interp_path = 'initial_path.traj'
        elif isinstance(images, str):
            interp_path = io.read(images, ':')
        else:
            raise(TypeError, 'You must include a set of images including an '
                  'optimized NEB path to start the optimization. The images '
                  'must be supplied as a list of Atoms or as the name of the '
                  'trajectory file including the set of images.')

        # GP calculator:
        self.trainingset = trainingset
        self.model_calculator = model_calculator
        self.max_train_data = max_train_data
        self.max_train_data_strategy = max_train_data_strategy
        if model_calculator is None:
            self.model_calculator = GPCalculator(
                            train_images=[],
                            prior=None,
                            fit_weight='update',
                            update_prior_strategy='fit',
                            weight=1.0, scale=0.4, noise=0.005,
                            update_hyperparams=False, batch_size=5,
                            bounds=None, kernel=None,
                            max_train_data_strategy=max_train_data_strategy,
                            max_train_data=max_train_data
                            )

        # Active Learning setup (Single-point calculations).
        self.step = 0
        self.function_calls = 0
        self.force_calls = 0
        self.ase_calc = calculator
        self.atoms = io.read(interp_path, '-1')

        self.constraints = self.atoms.constraints
        self.force_consistent = force_consistent
        self.use_previous_observations = use_previous_observations
        self.trajectory = trajectory

        # Create set of structure for AIDMin and AIDTS.
        images_path = io.read(interp_path, ':')

        # Initial minimum:
        atoms = io.read(interp_path, '1')
        atoms.set_calculator(self.ase_calc)
        atoms.positions = images[0].positions
        atoms.info['method_maxmin'] = 'min'
        self.list_atoms = [atoms]

        for i in range(1, len(images_path)-1):
            energy_L = images_path[i-1].get_potential_energy()
            energy_M = images_path[i].get_potential_energy()
            energy_R = images_path[i+1].get_potential_energy()
            if energy_L < energy_M and energy_R < energy_M:  # Maximum found.
                atoms = io.read(interp_path, '-1')
                atoms.set_calculator(self.ase_calc)
                atoms.positions = images_path[i].positions
                atoms.info['method_maxmin'] = 'max'
                self.list_atoms += [atoms]

            if energy_L > energy_M and energy_R > energy_M:  # Minimum found.
                atoms = io.read(interp_path, '-1')
                atoms.set_calculator(self.ase_calc)
                atoms.positions = images_path[i].positions
                atoms.info['method_maxmin'] = 'min'
                self.list_atoms += [atoms]

        # Final minimum.
        atoms = io.read(interp_path, '1')
        atoms.set_calculator(self.ase_calc)
        atoms.positions = images[-1].positions
        atoms.info['method_maxmin'] = 'min'
        self.list_atoms += [atoms]

        # Initialize training set
        trajectory_main = self.trajectory.split('.')[0]
        self.train = TrainingSet(trajectory_main + '_MEP.traj',
                                 use_previous_observations=False)

    def run(self, fmax=0.05, vector_length=0.7):

        """
        Executing run will start the NEB optimization process.

        Parameters
        ----------
        fmax : float
            Convergence criteria (in eV/Angstrom).
        vector_length: float
            Magnitude of the length vector for the Dimer.

        Returns
        -------
        Minimum Energy Path from the initial to the final states.

        """

        trajectory_main = self.trajectory.split('.')[0]
        trajectory_mep = trajectory_main + '_MEP.traj'

        if os.path.exists(trajectory_mep):
            os.remove(trajectory_mep)

        for i in self.list_atoms:
            if i.info['method_maxmin'] == 'min':
                self.atoms.set_calculator(self.ase_calc)
                self.atoms.positions = i.positions
                opt = AIDMin(
                    atoms=self.atoms, model_calculator=self.model_calculator,
                    trajectory=self.trajectory,
                    use_previous_observations=self.use_previous_observations,
                    max_train_data=self.max_train_data,
                    max_train_data_strategy=self.max_train_data_strategy,
                    trainingset=self.trainingset,
                    force_consistent=self.force_consistent
                    )
                opt.run(fmax=fmax)
                self.train.dump(atoms=self.atoms, method='mep')

            if i.info['method_maxmin'] == 'max':

                trajectory_main = self.trajectory.split('.')[0]
                last_min = io.read(trajectory_main + '_MEP.traj', -1)
                last_min.set_calculator(self.ase_calc)
                self.atoms.positions = i.positions

                opt = AIDTS(
                    atoms=last_min,
                    atoms_vector=self.atoms,
                    vector_length=vector_length,
                    model_calculator=self.model_calculator,
                    force_consistent=self.force_consistent,
                    trajectory=self.trajectory,
                    trainingset=self.trainingset,
                    max_train_data=self.max_train_data * 10,
                    max_train_data_strategy=self.max_train_data_strategy,
                    use_previous_observations=self.use_previous_observations
                    )
                opt.run(fmax=fmax)
                self.train.dump(atoms=last_min, method='mep')

        print_cite_aidmep()


@parallel_function
def print_cite_aidmep():
    msg = "\n" + "-" * 79 + "\n"
    msg += "You are using AIDMEP. Please cite: \n"
    msg += "[1] J. A. Garrido Torres, E. Garijo del Rio, V. Streibel "
    msg += "T. S. Choski, J. J. Mortensen, A. Urban, M. Bajdich "
    msg += "F. Abild-Pedersen, K. W. Jacobsen, and T. Bligaard. Submitted. \n"
    msg += "[2] E. Garijo del Rio, J. J. Mortensen and K. W. Jacobsen. "
    msg += "Phys. Rev. B 100, 104103."
    msg += "https://doi.org/10.1103/PhysRevB.100.104103. \n"
    msg += "[3] J. A. Garrido Torres, M. H. Hansen, P. C. Jennings, "
    msg += "J. R. Boes and T. Bligaard. Phys. Rev. Lett. 122, 156001. "
    msg += "https://doi.org/10.1103/PhysRevLett.122.156001 \n"
    msg += "-" * 79 + '\n'
    parprint(msg)
