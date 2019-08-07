import numpy as np
import copy
import time
from ase import io
from ase.atoms import Atoms
from ase.optimize.activelearning.gp.calculator import GPCalculator
from ase.neb import NEB
from ase.optimize import FIRE
from ase.optimize.activelearning.acquisition import acquisition
from ase.parallel import parprint, parallel_function
from ase.optimize.activelearning.io import get_fmax, dump_observation


class AIDNEB:

    def __init__(self, start, end, model_calculator=None, calculator=None,
                 interpolation='linear', n_images=0.25, k=None, mic=False,
                 neb_method='aseneb', remove_rotation_and_translation=False,
                 max_train_data=5, force_consistent=None,
                 max_train_data_strategy='nearest_observations',
                 trajectory='AIDNEB.traj', restart=False):

        """
        Artificial Intelligence-Driven Nudged Elastic Band (AID-NEB) algorithm.
        Optimize a NEB using a surrogate machine learning model [1,2].
        Potential energies and forces at a given position are
        supplied to the model calculator to build a modelled PES in an
        active-learning fashion. This surrogate relies on NEB theory to
        optimize the images along the path in the predicted PES. Once the
        predicted NEB is optimized the acquisition function collect a new
        observation based on the predicted energies and uncertainties of the
        optimized images.
        [1] J. A. Garrido Torres, M. H. Hansen, P. C. Jennings, J. R. Boes
        and T. Bligaard. Phys. Rev. Lett. 122, 156001.
        https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.122.156001
        [2] O. Koistinen, F. B. Dagbjartsdottir, V. Asgeirsson, A. Vehtari
        and H. Jonsson. J. Chem. Phys. 147, 152720.
        https://doi.org/10.1063/1.4986787
        [3] E. Garijo del Rio, J. J. Mortensen and K. W. Jacobsen.
        arXiv:1808.08588

        NEB Parameters
        --------------
        initial: Trajectory file (in ASE format) or Atoms object.
            Initial end-point of the NEB path.

        final: Trajectory file (in ASE format) or Atoms object.
            Final end-point of the NEB path.

        model_calculator: Model object.
            Model calculator to be used for predicting the potential energy
            surface. The default value is None which uses a GP model with the
            Squared Exponential Kernel and other default parameters. See
            *ase.calculator.gp.calculator* GPModel for default GP parameters.

        interpolation: string or Atoms list or Trajectory
            NEB interpolation.

            options:
                - 'linear' linear interpolation.
                - 'idpp'  image dependent pair potential interpolation.
                - Trajectory file (in ASE format) or list of Atoms.
                Trajectory or list of Atoms containing the images along the
                path.

        mic: boolean
            Use mic=True to use the Minimum Image Convention and calculate the
            interpolation considering periodic boundary conditions.

        n_images: int or float
            Number of images of the path. Only applicable if 'linear' or
            'idpp' interpolation has been chosen.
            options:
                - int: Number of images describing the NEB. The number of
                images include the two (initial and final) end-points of the
                NEB path.
                - float: Spacing of the images along the NEB. The number of
                images is therefore calculated as the length of the
                interpolated initial path divided by the spacing (Ang^-1).

        k: float or list
            Spring constant(s) in eV/Angstrom.

        neb_method: string
            NEB method as implemented in ASE. ('aseneb', 'improvedtangent'
            or 'eb'). See https://wiki.fysik.dtu.dk/ase/ase/neb.html.

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

        max_train_data: int
            Number of observations that will effectively be included in the
            model. See also *max_data_strategy*.

        max_train_data_strategy: string
            Strategy to decide the observations that will be included in the model.

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
        # Convert Atoms and list of Atoms to trajectory files.
        if isinstance(start, Atoms):
            io.write('initial.traj', start)
            initial = 'initial.traj'
        if isinstance(end, Atoms):
            io.write('final.traj', end)
            end = 'final.traj'
        interp_path = None
        if interpolation != 'idpp' and interpolation != 'linear':
            interp_path = interpolation
        if isinstance(interp_path, list):
            io.write('initial_path.traj', interp_path)
            interp_path = 'initial_path.traj'

        # NEB parameters.
        self.start = start
        self.end = end
        self.n_images = n_images
        self.mic = mic
        self.rrt = remove_rotation_and_translation
        self.neb_method = neb_method
        self.spring = k
        self.i_endpoint = io.read(self.start, '-1')
        self.i_endpoint.get_potential_energy(force_consistent=force_consistent)
        self.i_endpoint.get_forces()
        self.e_endpoint = io.read(self.end, '-1')
        self.e_endpoint.get_potential_energy(force_consistent=force_consistent)
        self.e_endpoint.get_forces()

        # Model calculator:
        self.model_calculator = model_calculator
        # Default GP Calculator parameters if not specified by the user.
        if model_calculator is None:
            self.model_calculator = GPCalculator(
                               train_images=[], scale=0.35, weight=2.0,
                               max_train_data_strategy=max_train_data_strategy,
                               max_train_data=max_train_data)

        # Active Learning setup (Single-point calculations).
        self.function_calls = 0
        self.force_calls = 0
        self.step = 0
        self.ase_calc = calculator
        self.atoms = io.read(self.start, '-1')

        self.constraints = self.atoms.constraints
        self.fc = force_consistent
        self.restart = restart
        self.trajectory = trajectory

        # Calculate the distance between the initial and final endpoints.
        d_start_end = np.linalg.norm(self.i_endpoint.positions.reshape(-1) -
                                     self.e_endpoint.positions.reshape(-1))
        # A) Create images using interpolation if user do defines a path.
        if interp_path is None:
            if isinstance(self.n_images, float):
                self.n_images = int(d_start_end/self.n_images) + 2
            if self. n_images <= 3:
                self.n_images = 3
            self.images = make_neb(self)

            # Guess spring constant (k) if not defined by the user.
            if self.spring is None:
                self.spring = (np.sqrt(self.n_images-1) / d_start_end)

            neb_interpolation = NEB(self.images, climb=False, k=self.spring,
                                    remove_rotation_and_translation=self.rrt)
            neb_interpolation.interpolate(method='linear',
                                          mic=self.mic)
            if interpolation == 'idpp':
                neb_interpolation = NEB(self.images, climb=True, k=self.spring,
                                        remove_rotation_and_translation=self.rrt)
                neb_interpolation.idpp_interpolate(optimizer=FIRE)

        # B) Alternatively, the user can manually decide the initial path.
        if interp_path is not None:
            images_path = io.read(interp_path, ':')
            first_image = images_path[0].get_positions().reshape(-1)
            last_image = images_path[-1].get_positions().reshape(-1)
            is_pos = self.i_endpoint.get_positions().reshape(-1)
            fs_pos = self.e_endpoint.get_positions().reshape(-1)
            if not np.array_equal(first_image, is_pos):
                images_path.insert(0, self.i_endpoint)
            if not np.array_equal(last_image, fs_pos):
                images_path.append(self.e_endpoint)
            self.n_images = len(images_path)
            self.images = make_neb(self, images_interpolation=images_path)

        # Guess spring constant (k) if not defined by the user.
        if self.spring is None:
            self.spring = (np.sqrt(self.n_images-1) / d_start_end)

        trajectory_main = self.trajectory.split('.')[0]
        self.trajectory_observations = trajectory_main + '_observations.traj'
        self.trajectory_candidates = trajectory_main + '_candidates.traj'

        # Start by saving the initial and final states.
        dump_observation(atoms=self.i_endpoint, method='neb',
                         filename=self.trajectory_observations,
                         restart=self.restart)
        self.restart = True  # Switch on active learning.
        dump_observation(atoms=self.e_endpoint, method='neb',
                         filename=self.trajectory_observations,
                         restart=self.restart)

    def run(self, fmax=0.05, unc_convergence=0.05, ml_steps=50, max_step=2.0):

        """
        Executing run will start the NEB optimization process.

        Parameters
        ----------
        fmax : float
            Convergence criteria (in eV/Angstrom).

        unc_convergence: float
            Maximum uncertainty for convergence (in eV). The algorithm's
            convergence criteria will not be satisfied if the uncertainty
            on any of the NEB images in the predicted path is above this
            threshold.

        ml_steps: int
            Maximum number of steps for the NEB optimization on the
            modelled potential energy surface.

        max_step: float
            Safe control parameter. This parameter controls whether the
            optimization of the NEB will be performed in the modelled
            potential energy surface. If the uncertainty of the NEB lies
            above the 'max_step' threshold the NEB won't be optimized and
            the image with maximum uncertainty is evaluated. This prevents
            exploring very uncertain regions which can lead to probe
            unrealistic structures.
        """

        while True:

            # 1. Gather observations in every iteration.
            # This serves to restart from a previous (and/or parallel) runs.
            train_images = io.read(self.trajectory_observations, ':')

            # 2. Prepare the model calculator (train and attach to images).
            calc = copy.deepcopy(self.model_calculator)

            # Detach calculator from the prev. optimized images (speed up).
            for i in self.images:
                i.set_calculator(None)

            # Train only one process at the time.
            calc.update_train_data(train_images,
                                   test_images=copy.deepcopy(self.images))

            # Attach the calculator (already trained) to each image.
            for i in self.images:
                i.set_calculator(copy.deepcopy(calc))

            # 3. Optimize the NEB in the predicted PES.
            # Get path uncertainty for selecting within NEB or CI-NEB.
            predictions = get_neb_predictions(self.images)
            neb_pred_uncertainty = predictions['uncertainty']

            # Climbing image NEB mode is risky when the model is trained
            # with a few data points. Switch on climbing image (CI-NEB) only
            # when the uncertainty of the NEB is low.
            climbing_neb = False
            if np.max(neb_pred_uncertainty) <= unc_convergence:
                parprint('Climbing image is now activated.')
                climbing_neb = True
            ml_neb = NEB(self.images, climb=climbing_neb,
                         method=self.neb_method, k=self.spring,
                         remove_rotation_and_translation=self.rrt)
            neb_opt = FIRE(ml_neb, trajectory=self.trajectory)

            # Safe check to optimize the images.
            if np.max(neb_pred_uncertainty) <= max_step:
                neb_opt.run(fmax=(fmax * 0.80), steps=ml_steps)

            # 4. Get predicted energies and uncertainties of the NEB images.
            predictions = get_neb_predictions(self.images)
            neb_pred_energy = predictions['energy']
            neb_pred_uncertainty = predictions['uncertainty']

            # 5. Print output.
            max_e = np.max(neb_pred_energy)
            pbf = max_e - self.i_endpoint.get_potential_energy(
                                                      force_consistent=self.fc)
            pbb = max_e - self.e_endpoint.get_potential_energy(
                                                      force_consistent=self.fc)
            parprint("-" * 26)
            parprint('Step:', self.function_calls)
            parprint('Time:', time.strftime("%m/%d/%Y, %H:%M:%S",
                                            time.localtime()))
            parprint('Predicted barrier (-->):', pbf)
            parprint('Predicted barrier (<--):', pbb)
            parprint('Max. uncertainty:', np.max(neb_pred_uncertainty))
            parprint('Number of images:', len(self.images))
            parprint("fmax:", get_fmax(train_images[-1]))
            parprint("-" * 26)

            # 6. Check convergence.
            # The uncertainty of all NEB images must be below the
            # *unc_convergence* threshold and the climbing image must
            # satisfy the *fmax* convergence criteria.
            if self.step > 1 and get_fmax(train_images[-1]) <= fmax:
                parprint('A saddle point was found.')
                if np.max(neb_pred_uncertainty[1:-1]) < unc_convergence:
                    io.write(self.trajectory, self.images)
                    parprint('Uncertainty of the images above threshold.')
                    parprint('NEB converged.')
                    parprint('The NEB path can be found in:', self.trajectory)
                    msg = "Visualize the last path using 'ase gui "
                    msg += self.trajectory
                    parprint(msg)
                    break

            # 7. Convergence criteria not satisfied?
            #  Then, select a new geometry to evaluate (acquisition function):

            # Candidates are the optimized NEB images in the predicted PES.
            candidates = self.images

            # This acquisition function has been tested in Ref. [1].
            if np.max(neb_pred_uncertainty) > unc_convergence:
                sorted_candidates = acquisition(train_images=train_images,
                                                candidates=candidates,
                                                mode='uncertainty',
                                                objective='max')
            else:
                sorted_candidates = acquisition(train_images=train_images,
                                                candidates=candidates,
                                                mode='ucb',
                                                objective='max')

            # Select the best candidate.
            best_candidate = sorted_candidates.pop(0)

            # Save the other candidates for multi-task optimization.
            io.write(self.trajectory_candidates, sorted_candidates)

            # 8. Evaluate the target function and save it in *observations*.
            self.atoms.positions = best_candidate.get_positions()
            self.atoms.set_calculator(self.ase_calc)
            self.atoms.get_potential_energy(force_consistent=self.fc)
            self.atoms.get_forces()
            dump_observation(atoms=self.atoms, method='neb',
                             filename=self.trajectory_observations,
                             restart=self.restart)
            self.function_calls += 1
            self.force_calls += 1
            self.step += 1
        print_cite_neb()


@parallel_function
def make_neb(self, images_interpolation=None):
    """
    Creates a NEB path from a set of images.
    """
    imgs = [self.i_endpoint[:]]
    for i in range(1, self.n_images-1):
        image = self.i_endpoint[:]
        if images_interpolation is not None:
            image.set_positions(images_interpolation[i].get_positions())
        image.set_constraint(self.constraints)
        imgs.append(image)
    imgs.append(self.e_endpoint[:])
    return imgs


@parallel_function
def get_neb_predictions(images):
    """
    Collects the predicted energy values and uncertainties of the NEB images.
    """
    neb_pred_energy = []
    neb_pred_unc = []
    for i in images:
        neb_pred_energy.append(i.get_potential_energy())
        unc = i.get_calculator().results['uncertainty']
        neb_pred_unc.append(unc)
    neb_pred_unc[0] = 0.0
    neb_pred_unc[-1] = 0.0
    predictions = {'energy': neb_pred_energy, 'uncertainty': neb_pred_unc}
    return predictions


@parallel_function
def print_cite_neb():
    msg = "\n" + "-" * 79 + "\n"
    msg += "You are using AID-NEB. Please cite: \n"
    msg += "[1] J. A. Garrido Torres, M. H. Hansen, P. C. Jennings, "
    msg += "J. R. Boes and T. Bligaard. Phys. Rev. Lett. 122, 156001. "
    msg += "https://doi.org/10.1103/PhysRevLett.122.156001 \n"
    msg += "[2] O. Koistinen, F. B. Dagbjartsdottir, V. Asgeirsson, A. Vehtari"
    msg += " and H. Jonsson. J. Chem. Phys. 147, 152720. "
    msg += "https://doi.org/10.1063/1.4986787 \n"
    msg += "[3] E. Garijo del Rio, J. J. Mortensen and K. W. Jacobsen. "
    msg += "arXiv:1808.08588. https://arxiv.org/abs/1808.08588v1. \n"
    msg += "-" * 79 + '\n'
    parprint(msg)
