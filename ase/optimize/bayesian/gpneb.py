import numpy as np
import copy
import time
from ase.parallel import parprint, parallel_function
from ase import io
from ase.atoms import Atoms
from ase.neb import NEB
from ase.optimize import MDMin
from ase.geometry import distance
from ase.optimize.bayesian.io import print_cite_neb, dump_experiences, get_fmax
from ase.optimize.bayesian.model import GPModel, create_mask


class GPNEB:

    def __init__(self, start, end, calculator=None, model=None,
                 interpolation='linear', n_images=0.25, k=None, mic=False,
                 neb_method='aseneb', remove_rotation_and_translation=False,
                 force_consistent=None):

        """
        Machine Learning Nudged elastic band (NEB).
        Optimize a NEB using a surrogate machine learning model [1,2].
        Potential energies and forces information are used to build a
        predicted PES via Gaussian Process (GP) regression. The surrogate
        relies on NEB theory to optimize the images along the path in the
        predicted PES. Once the predicted NEB is optimized the acquisition
        function collect a new experience based on the predicted energies
        and uncertainties of the optimized images.
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
        start: Trajectory file (in ASE format) or Atoms object.
            Initial end-point of the NEB path.

        end: Trajectory file (in ASE format) or Atoms object.
            Final end-point of the NEB path.

        model: Model object.
            Model to be used for predicting the potential energy surface.
            The default is None which uses a GP model with the Squared
            Exponential Kernel and other default parameters. See
            ase.optimize.bayesian.model GPModel for default GP parameters.

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
            Spring constant(s) in eV/Ang.

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

        """
        # Convert Atoms and list of Atoms to trajectory files.
        if isinstance(start, Atoms):
            io.write('initial.traj', start)
            start = 'initial.traj'
        if isinstance(end, Atoms):
            io.write('final.traj', end)
            end = 'final.traj'
        interp_path = None
        if interpolation != 'idpp' and interpolation != 'linear':
            interp_path = interpolation
        if isinstance(interp_path, list):
            io.write('initial_path.traj', interp_path)
            interp_path = 'initial_path.traj'

        # Predictive model parameters:
        self.model = model
        if model is None:
            self.model = GPModel()

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
        self.e_endpoint = io.read(self.end, '-1')
        self.e_endpoint.get_potential_energy(force_consistent=force_consistent)

        # General setup.
        self.ase_calc = calculator
        self.atoms_template = io.read(self.start, '-1')
        self.constraints = self.atoms_template.constraints

        # Optimization.
        self.function_calls = 0
        self.force_calls = 0
        self.force_consistent = force_consistent

        # Calculate the distance between the initial and final endpoints.
        d_start_end = distance(self.i_endpoint, self.e_endpoint)

        # A) Create images using interpolation if user do defines a path.
        if interp_path is None:
            if isinstance(self.n_images, float):
                self.n_images = int(d_start_end/self.n_images)
            if self. n_images <= 3:
                self.n_images = 3
            self.images = make_neb(self)
            neb_interpolation = NEB(self.images)
            neb_interpolation.interpolate(method=interpolation, mic=self.mic)

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

        # Initialize the set of evaluated atoms structures with the initial
        # and final endpoints.
        self.images_pool = [self.i_endpoint, self.e_endpoint]

        # Create mask to hide atoms that do not participate in the model:
        if self.constraints is not None:
            self.atoms_mask = create_mask(self.atoms_template)

    def run(self, fmax=0.05, unc_convergence=0.05, dt=0.05, ml_steps=200,
            max_step=0.5, trajectory='GPNEB.traj', restart=False):

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

        dt : float
            dt parameter for MDMin.

        ml_steps: int
            Maximum number of steps for the MDMin/NEB optimization on the GP
            predicted potential energy surface.

        max_step: float
            Safe control parameter. This parameter controls whether the
            optimization of the NEB will be performed in the predicted
            potential energy surface. If the uncertainty of the NEB lies
            above the 'max_step' threshold the NEB won't be optimized and
            the image with maximum uncertainty is evaluated. This prevents
            exploring very uncertain regions which can lead to probe
            unrealistic structures.

        trajectory: string
            Filename to store the predicted NEB paths.
                Additional information:
                - Energy uncertain: The energy uncertainty in each image can be
                  accessed in image.info['uncertainty'].

        restart: bool
            A *trajectory_experiences.traj* file is automatically generated
            which contains the experiences collected by the surrogate. If
            *restart* is True and a *trajectory_experiences.traj* file is
            found in the working directory it will be used to continue the
            optimization from previous run(s). In order to start the
            optimization from scratch *restart* should be set to False or
            alternatively the *trajectory_experiences.traj* file must be
            deleted.

        Returns
        -------
        Minimum Energy Path from the initial to the final states.

        """

        while True:

            # 1. Experiences. Restart from a previous (and/or parallel) runs.
            start = time.time()
            dump_experiences(images=self.images_pool,
                             filename=trajectory, restart=restart)
            if restart is True:
                self.images_pool = io.read(trajectory.split('.')[0] +
                                           '_experiences.traj', ':')
            end = time.time()
            parprint('Time reading and writing atoms images to build a model:',
                     end-start)

            # 2. Build a ML model from Atoms images (i.e. images_pool).
            start = time.time()
            self.model.extract_features(train_images=self.images_pool,
                                        atoms_mask=self.atoms_mask,
                                        force_consistent=self.force_consistent)
            end = time.time()
            parprint('Elapsed time featurizing data:', end-start)

            start = time.time()
            self.model.train_model()
            end = time.time()
            parprint('Elapsed time training the model:', end-start)

            # 3. Get predictions for the geometries to be tested.
            self.model.get_images_predictions(test_images=self.images,
                                              get_uncertainty=True)
            neb_pred_uncertainty = self.model.pred_uncertainty
            # Initial and final end-points with zero uncertainty.
            neb_pred_uncertainty[0] = 0.0
            neb_pred_uncertainty[-1] = 0.0

            # Climbing image NEB mode is risky when the model is trained
            # with a few data points. Switch on climbing image (CI-NEB) only
            # when the uncertainty of the NEB is low.
            climbing_neb = False
            if np.max(neb_pred_uncertainty) <= unc_convergence:
                parprint('Climbing image is now activated.')
                climbing_neb = True

            # Switch off calculating uncertainty to make faster predictions.
            for i in self.images:
                i.get_calculator().get_variance = False

            ml_neb = NEB(self.images, climb=climbing_neb,
                         method=self.neb_method, k=self.spring)
            neb_opt = MDMin(ml_neb, dt=dt, trajectory=trajectory)

            # Safe check to optimize the images.
            if np.max(neb_pred_uncertainty) <= max_step:
                start = time.time()
                parprint('Optimizing NEB in the model potential...')
                neb_opt.run(fmax=(fmax * 0.80), steps=ml_steps)
                parprint('Optimized NEB in the model potential.')
                end = time.time()
                parprint('Elapsed time optimizing predicted NEB:', end-start)

            # 4. Print output predictions.
            self.model.get_images_predictions(test_images=self.images,
                                              get_uncertainty=True)
            neb_pred_uncertainty = self.model.pred_uncertainty
            # Initial and final end-points with zero uncertainty.
            neb_pred_uncertainty[0] = 0.0
            neb_pred_uncertainty[-1] = 0.0
            neb_pred_energy = self.model.pred_energy

            # 7. Print output.
            max_e = np.max(neb_pred_energy)
            pbf = max_e - self.i_endpoint.get_potential_energy()
            pbb = max_e - self.e_endpoint.get_potential_energy()
            msg = "--------------------------------------------------------"
            parprint(msg)
            parprint('Step:', self.function_calls)
            parprint('Time:', time.strftime("%m/%d/%Y, %H:%M:%S",
                                            time.localtime()))
            parprint('Predicted barrier (-->):', pbf)
            parprint('Predicted barrier (<--):', pbb)
            parprint('Max. uncertainty:', np.max(neb_pred_uncertainty))
            parprint('Number of images:', len(self.images))
            parprint("fmax:", get_fmax(self.images_pool[-1]))
            msg = "--------------------------------------------------------\n"
            parprint(msg)

            # 5. Check convergence.
            # Max.forces and NEB images uncertainty must be below *fmax* and
            # *unc_convergence* thresholds.
            last_fmax = get_fmax(self.images_pool[-1])
            if len(self.images_pool) > 2 and last_fmax <= fmax:
                parprint('A saddle point was found.')
                if np.max(neb_pred_uncertainty[1:-1]) < unc_convergence:
                    parprint('Uncertainty of the images above threshold.')
                    parprint('NEB converged.')
                    parprint('The converged NEB path can be found in:',
                             trajectory)
                    msg = "Visualize the last path using 'ase gui "
                    msg += trajectory + "@-"
                    msg += str(self.n_images) + ":'"
                    parprint(msg)
                    break

            # 6. Collect predicted values from the optimized ML NEB.
            self.model.get_images_predictions(test_images=self.images)
            neb_pred_energy = self.model.pred_energy
            neb_pred_uncertainty = self.model.pred_uncertainty

            # 7. Select next point to train (acquisition function):
            e_plus_unc_pred_neb = np.array(neb_pred_energy)
            e_plus_unc_pred_neb += np.array(neb_pred_uncertainty)
            img_max_unc = self.images[np.argmax(neb_pred_uncertainty)]
            img_max_e_plus_unc = self.images[np.argmax(e_plus_unc_pred_neb)]

            # Target image with max. uncertainty until we reach the
            # unc_convergence threshold.
            if np.max(neb_pred_uncertainty) > unc_convergence:
                positions_to_evaluate = img_max_unc.positions
            # Target top energy images when the unc_convergence threshold
            # has been satisfied.
            else:
                positions_to_evaluate = img_max_e_plus_unc.positions

            # 8. Evaluate the target function and add it to the pool of
            # evaluated atoms structures.
            parprint('Performing evaluation on the real landscape...')
            eval_atoms = Atoms(self.atoms_template,
                               positions=positions_to_evaluate,
                               calculator=self.ase_calc)
            eval_atoms.get_potential_energy(force_consistent=self.force_consistent)
            self.images_pool += [copy.deepcopy(eval_atoms)]
            self.function_calls += 1
            self.force_calls += 1
            parprint('Single-point calculation finished.')

        # Print final output when the surrogate is converged.
        print_cite_neb()
        parprint('The converged NEB path can be found in:', trajectory)
        msg = "Visualize the last path using 'ase gui " + trajectory + "@-"
        msg += str(self.n_images) + ":'"
        parprint(msg)


@parallel_function
def make_neb(self, images_interpolation=None):
    """
    Creates a NEB from a set of images.
    """
    imgs = [self.i_endpoint[:]]
    for i in range(1, self.n_images-1):
        image = self.i_endpoint.copy()
        if images_interpolation is not None:
            image.set_positions(images_interpolation[i].get_positions())
        image.set_constraint(self.constraints)
        imgs.append(image)
    imgs.append(self.e_endpoint[:])
    return imgs
