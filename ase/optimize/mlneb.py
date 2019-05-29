import numpy as np
import time
from ase.neb import NEB
from ase.io import read, write, Trajectory
from ase.atoms import Atoms
from ase.geometry import distance
from ase.parallel import parprint, rank, parallel_function
from ase.calculators.calculator import Calculator, all_changes
from ase.optimize import MDMin
from scipy.linalg import solve_triangular
from ase.optimize.gpmin.kernel import SquaredExponential
from ase.optimize.gpmin.gp import GaussianProcess
from ase.optimize.gpmin.prior import ConstantPrior


class MLNEB(GaussianProcess):

    def __init__(self, start, end, ase_calc=None,
                 restart='evaluated_structures.traj',
                 interpolation='linear', n_images=0.25, k=None, mic=False,
                 neb_method='aseneb', remove_rotation_and_translation=False,
                 prior=None, update_prior_strategy='maximum',
                 weight=1.0, scale=0.35, noise=0.005,
                 force_consistent=None,
                 update_hyperparams=False, batch_size=5, bounds=None):

        """
        Machine Learning Nudged elastic band (NEB).
        Optimize NEB using the ML-NEB algorithm [1, 2], which uses
        both potential energies and forces information to build a PES
        via Gaussian Process (GP) regression and then optimizes a NEB.
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

        restart: string or None
            Name of trajectory file (in ASE format) to store/read the data
            that can be used to restart the calculation. If None the
            optimizer won't save a restart file and hence the calculation
            can't be restarted. If a file is defined the optimizer saves a
            trajectory which the training data is saved. If the calculation
            is restarted the optimizer reads the restart file and continues
            from there.

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

        ase_calc: ASE calculator Object.
            ASE calculator.
            See https://wiki.fysik.dtu.dk/ase/ase/calculators/calculators.html

        force_consistent: boolean or None
            Use force-consistent energy calls (as opposed to the energy
            extrapolated to 0 K). By default (force_consistent=None) uses
            force-consistent energies if available in the calculator, but
            falls back on force_consistent=False if not.

        GP Parameters
        --------------
        prior: Prior object or None
            Prior for the GP regression of the PES surface
            See ase.optimize.gpmin.prior
            If *Prior* is None, then it is set as the
            ConstantPrior with the constant being updated
            using the update_prior_strategy specified as a parameter

        weight: float
            Pre-exponential factor of the Squared Exponential kernel.
            If *update_hyperparams* is False, changing this parameter
            has no effect on the dynamics of the algorithm.

        scale: float
            Scale of the Squared Exponential Kernel

        noise: float
            Regularization parameter for the Gaussian Process Regression.

        update_prior_strategy: string
            Strategy to update the constant from the ConstantPrior
            when more data is collected. It does only work when
            Prior = None

            options:
                'maximum': update the prior to the maximum sampled energy
                'init' : fix the prior to the initial energy
                'average': use the average of sampled energies as prior

        update_hyperparams: boolean
            Update the scale of the Squared exponential kernel
            every batch_size-th iteration by maximizing the
            marginal likelihood.

        batch_size: int
            Number of new points in the sample before updating
            the hyperparameters.
            Only relevant if the optimizer is executed in update
            mode: (update = True)

        bounds: float, 0<bounds<1
            Set bounds to the optimization of the hyperparameters.
            Let t be a hyperparameter. Then it is optimized under the
            constraint (1-bound)*t_0 <= t <= (1+bound)*t_0
            where t_0 is the value of the hyperparameter in the previous
            step. If bounds is None, no constraints are set in the
            optimization of the hyperparameters.
        """

        # Convert Atoms and list of Atoms to trajectory files.
        if isinstance(start, Atoms):
            write('initial.traj', start)
            start = 'initial.traj'
        if isinstance(end, Atoms):
            write('final.traj', end)
            end = 'final.traj'
        interp_path = None
        if interpolation != 'idpp' and interpolation != 'linear':
            interp_path = interpolation
        if isinstance(interp_path, list):
            write('initial_path.traj', interp_path)
            interp_path = 'initial_path.traj'

        # NEB parameters.
        self.i_endpoint = read(start, '-1')
        self.e_endpoint = read(end, '-1')
        self.n_images = n_images
        self.mic = mic
        self.rrt = remove_rotation_and_translation
        self.neb_method = neb_method
        self.spring = k

        # General setup.
        self.ase_calc = ase_calc
        self.atoms_template = read(start, '-1')
        self.restart = restart

        # Optimization.
        self.function_calls = 0
        self.force_calls = 0
        self.force_consistent = force_consistent

        # Mask training features that do not participate in the optimization.
        self.constraints = self.atoms_template.constraints
        self.index_mask = create_mask(self.atoms_template,
                                      self.constraints)

        # Checks.
        msg = 'Error: Initial structure for the NEB was not provided.'
        assert start is not None, msg
        msg = 'Error: Final structure for the NEB was not provided.'
        assert end is not None, msg
        msg = 'ASE calculator not provided (see "ase_calc" flag).'
        assert self.ase_calc, msg

        # Gaussian Process parameters.
        self.x_list = []
        self.y_list = []
        self.nbatch = batch_size
        self.strategy = update_prior_strategy
        self.update_hp = update_hyperparams
        self.eps = bounds

        if prior is None:
            self.update_prior = True
            prior = ConstantPrior(constant=None)

        else:
            self.update_prior = False

        # Set kernel and prior.
        kernel = SquaredExponential()
        GaussianProcess.__init__(self, prior, kernel)

        # Set initial hyperparameters.
        self.set_hyperparams(np.array([weight, scale, noise]))

        # Calculate the distance between the initial and final endpoints.
        d_start_end = distance(self.i_endpoint, self.e_endpoint)

        # A) Create images using interpolation if user do defines a path.
        if interp_path is None:
            if isinstance(self.n_images, float):
                self.n_images = int(d_start_end/self.n_images)
            if self. n_images <= 3:
                self.n_images = 3
            self.images = create_ml_neb(self)
            neb_interpolation = NEB(self.images)
            neb_interpolation.interpolate(method=interpolation, mic=self.mic)

        # B) Alternatively, the user can manually decide the initial path.
        if interp_path is not None:
            images_path = read(interp_path, ':')
            first_image = images_path[0].get_positions().reshape(-1)
            last_image = images_path[-1].get_positions().reshape(-1)
            is_pos = self.i_endpoint.get_positions().reshape(-1)
            fs_pos = self.e_endpoint.get_positions().reshape(-1)
            if not np.array_equal(first_image, is_pos):
                images_path.insert(0, self.i_endpoint)
            if not np.array_equal(last_image, fs_pos):
                images_path.append(self.e_endpoint)
            self.n_images = len(images_path)
            self.images = create_ml_neb(self, images_interpolation=images_path)

        # Guess spring constant (k) if not defined by the user.
        if self.spring is None:
            self.spring = (np.sqrt(self.n_images-1) / d_start_end)

        # Initialize the set of evaluated atoms structures.
        self.atoms_pool = [self.i_endpoint, self.e_endpoint]

        # Include previous calculations to the set of evaluated structures.
        if self.restart is not None:
            try:
                self.atoms_pool += read(self.restart, ':')
            except Exception:
                pass

    def run(self, fmax=0.05, unc_convergence=0.05, dt=0.05, ml_steps=200,
            max_step=0.5, trajectory='ML-NEB.traj'):

        """
        Executing run will start the NEB optimization process.

        Parameters
        ----------
        fmax : float
            Convergence criteria (in eV/Angs).

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
            Filename to store the predicted NEB paths. Note: The energy
            uncertainty in each image can be accessed in image.info[
            'uncertainty'].

        Returns
        -------
        Minimum Energy Path from the initial to the final states.

        """

        # 1. Initialize. Gather GP data.
        for i in self.atoms_pool:
            update_gp_data(self, atoms=i)

        while True:

            # 3. Train model.
            # Mask the positions and forces of the atoms that do not
            # participate in the surrogate (e.g. fixed atoms):
            train_x = self.x_list[:]
            train_y = self.y_list[:]

            for i in range(0, len(train_x)):
                train_x[i] = train_x[i][self.index_mask].reshape(-1)
                train_y[i] = np.concatenate(([train_y[i][0]], train_y[i][
                                             1:][self.index_mask].reshape(-1)))

            # 4. Build GP model.
            parprint('Training GP. Number of traning points:', len(train_y))
            self.train(np.array(train_x), np.array(train_y))

            # Optimize the hyperparameters (optional).
            if self.update_hp and self.function_calls % self.nbatch == 0 and self.function_calls != 0:
                parprint('Optimizing GP hyperparameters...')
                ratio = self.noise / self.kernel.weight
                try:
                    self.fit_hyperparameters(np.asarray(train_x),
                                             np.asarray(train_y),
                                             eps=self.eps)
                except Exception:
                    pass

                else:
                    # Keeps the ratio between noise and weight fixed.
                    self.noise = ratio * self.kernel.weight

            # 5. Get predictions for the geometries to be tested.
            predictions = get_neb_predictions(self)
            list_pred_e = predictions['pred_energy']
            list_pred_u = predictions['pred_uncertainty']
            max_unc_neb = np.max(np.abs(list_pred_u))

            # 6. Check whether converged. Predictions and fmax below threshold.
            if self.function_calls != 0 and get_fmax(self.atoms_pool[-1]) <= fmax:
                parprint('A saddle point was found.')
                if max_unc_neb <= unc_convergence:
                    parprint('Uncertainty of the images below threshold.')
                    parprint('ML-NEB converged.')
                    break

            # 7. Print output.
            pbf = np.max(list_pred_e) - self.i_endpoint.get_potential_energy()
            pbb = np.max(list_pred_e) - self.e_endpoint.get_potential_energy()
            msg = "--------------------------------------------------------"
            parprint(msg)
            parprint('Step:', self.function_calls)
            parprint('Time:', time.strftime("%m/%d/%Y, %H:%M:%S",
                                            time.localtime()))
            parprint('Predicted barrier (-->):', pbf)
            parprint('Predicted barrier (<--):', pbb)
            parprint('Max. uncertainty:', np.max(list_pred_u))
            parprint('Average uncertainty:', np.mean(list_pred_u))
            parprint('Number of images:', len(self.images))
            if self.function_calls != 0:
                parprint("fmax:", get_fmax(self.atoms_pool[-1]))
            msg = "--------------------------------------------------------\n"
            parprint(msg)

            # 8. Not converged? Optimize a NEB in the predicted potential.

            # Attach GP calculator to the images.
            self.images = create_ml_neb(self, images_interpolation=self.images)

            # Perform NEB in the predicted landscape.

            # Decide whether to activate CI-NEB or perform standard NEB.
            ml_neb = NEB(self.images, climb=False,
                         method=self.neb_method,
                         remove_rotation_and_translation=self.rrt,
                         k=self.spring)

            # Decide whether to activate CI-NEB or perform standard NEB.
            if max_unc_neb <= unc_convergence:
                parprint('CI-NEB. Climbing image on.')
                ml_neb = NEB(self.images, climb=True,
                             method=self.neb_method,
                             k=self.spring)
            neb_opt = MDMin(ml_neb, dt=dt, trajectory=trajectory)

            # Perform NEB optimization if the path is not highly uncertain.
            if self.function_calls != 0 and max_unc_neb <= max_step:
                parprint('Optimizing NEB in the GP potential...')
                neb_opt.run(fmax=(fmax * 0.80), steps=ml_steps)
                parprint('Optimized NEB in the GP potential.')

            # 9. Gather output predictions for the acquisition function.
            predictions = get_neb_predictions(self)
            list_pred_r = predictions['pred_positions']
            list_pred_e = predictions['pred_energy']
            list_pred_u = predictions['pred_uncertainty']
            max_unc_neb = np.max(np.abs(list_pred_u))

            # 10. Acquisition function. Decide the structure to calculate.
            pred_plus_unc = np.array(list_pred_e) + np.array(list_pred_u)

            # Select image with highest uncertainty.
            if max_unc_neb > unc_convergence:
                parprint('Evaluating the image with max. uncertainty...')
                max_score = np.argmax(list_pred_u)
            # Target image with max. predicted value when uncertainty is low.
            else:
                parprint('Evaluating climbing image...')
                max_score = np.argmax(pred_plus_unc)

            interesting_r = list_pred_r[int(max_score)]

            # 11. Add new point to the pool of atoms and update GP data.
            eval_structure = self.atoms_template
            eval_structure.positions = interesting_r.reshape(-1, 3)
            eval_structure.set_calculator(self.ase_calc)
            eval_structure.get_potential_energy(force_consistent=self.force_consistent)
            self.atoms_pool += [eval_structure]
            update_gp_data(self, atoms=self.atoms_pool[-1])
            self.function_calls += 1
            self.force_calls += 1

        # Print final output when converged.
        print_cite()
        parprint('The converged NEB path can be found in:', trajectory)
        msg = "Visualize the last path using 'ase gui " + trajectory + "@-"
        msg += str(self.n_images) + ":'"
        parprint(msg)


@parallel_function
def create_ml_neb(self, images_interpolation=None):
    """
    Generates a set atomic structure images representing the elastic band
    and attaches the GP calculator to the moving images.

    """
    imgs = [self.i_endpoint]
    for i in range(1, self.n_images-1):
        image = self.i_endpoint.copy()
        image.set_calculator(GPCalculator(self))
        if images_interpolation is not None:
            image.set_positions(images_interpolation[i].get_positions())
        image.set_constraint(self.constraints)
        imgs.append(image)
    imgs.append(self.e_endpoint)
    return imgs


@parallel_function
def update_gp_data(self, atoms):
    """
    Collect positions, energy and forces from a given atoms structure and
    update the GP training set, the prior and hyperparameters.
    """

    # Get information from the Atoms object.
    r = atoms.get_positions().reshape(-1)
    e = atoms.get_potential_energy(force_consistent=self.force_consistent)
    f = atoms.get_forces()

    # Update the training set.
    self.x_list.append(r)
    f = f.reshape(-1)
    y = np.append(np.array(e).reshape(-1), -f)
    self.y_list.append(y)

    # Set/update the constant for the prior.
    if self.update_prior:
        if self.strategy == 'average':
            av_e = np.mean(np.array(self.y_list)[:, 0])
            self.prior.set_constant(av_e)
        elif self.strategy == 'maximum':
            max_e = np.max(np.array(self.y_list)[:, 0])
            self.prior.set_constant(max_e)
        elif self.strategy == 'init':
            self.prior.set_constant(e)
            self.update_prior = False

    # Dump structure to file:
    if rank == 0 and self.restart is not None:
        dump_atoms(atoms, filename=self.restart)


@parallel_function
def get_neb_predictions(self):
    list_pred_r = []
    list_pred_e = []
    list_pred_u = []
    for i in self.images:
        i.set_calculator(GPCalculator(self, get_variance=True))
        list_pred_e.append(i.get_potential_energy(force_consistent=self.force_consistent))
        list_pred_r.append(i.get_positions().reshape(-1))
        uncertainty = i.get_calculator().results['uncertainty']
        if uncertainty < 0.0:
            uncertainty = 0.0
        list_pred_u.append(uncertainty)
        i.info['uncertainty'] = uncertainty
    list_pred_u[0] = 0.0  # Initial end-point.
    list_pred_u[-1] = 0.0  # Final end-point.

    return {'pred_positions':list_pred_r, 'pred_energy':list_pred_e,
            'pred_uncertainty':list_pred_u}


@parallel_function
def dump_atoms(atoms, filename):
    """
    Dumps Atoms structure into a pool of evaluated structures. This is
    exclusively activated if restart is not None. The function works
    together with the update_gp_data function.
    """

    # Initialize if previous data is not found.
    try:
        read(filename, ':')
    except Exception:
        Trajectory(filename, 'w', atoms).write()
    prev_atoms = read(filename, ':')

    # Append to the file (avoids duplicates).
    if atoms not in prev_atoms:
        parprint('Updating the restart file...')
        Trajectory(filename, 'a', atoms).write()
    else:
        parprint('Atoms object found on file (duplicate).')


@parallel_function
def get_fmax(atoms):
    """
    Returns fmax for a given atoms structure.
    """
    forces = atoms.get_forces()
    return np.sqrt((forces**2).sum(axis=1).max())

@parallel_function
def print_cite():
    msg = "-----------------------------------------------------------"
    msg += "-----------------------------------------------------------\n"
    msg += "You are using ML-NEB. Please cite: \n"
    msg += "[1] J. A. Garrido Torres, M. H. Hansen, P. C. Jennings, "
    msg += "J. R. Boes and T. Bligaard. Phys. Rev. Lett. 122, 156001. "
    msg += "https://doi.org/10.1103/PhysRevLett.122.156001 \n"
    msg += "[2] O. Koistinen, F. B. Dagbjartsdottir, V. Asgeirsson, A. Vehtari"
    msg += " and H. Jonsson. J. Chem. Phys. 147, 152720. "
    msg += "https://doi.org/10.1063/1.4986787 \n"
    msg += "[3] E. Garijo del Rio, J. J. Mortensen and K. W. Jacobsen. "
    msg += "arXiv:1808.08588. https://arxiv.org/abs/1808.08588v1. \n"
    msg += "-----------------------------------------------------------"
    msg += "-----------------------------------------------------------"
    parprint(msg)


@parallel_function
def create_mask(atoms, constraints):
    """
    Creates a mask with the index of the atoms' coordinates that are relaxed.
    Avoids training a GP process with elements that have no correlation.
    """
    m = np.ones_like(atoms.positions, dtype=bool)
    for i in range(0, len(constraints)):
        try:
            m[constraints[i].__dict__['a']] = ~(constraints[i].__dict__['mask'])
        except Exception:
            pass
        try:
            m[constraints[i].__dict__['index']] = False
        except Exception:
            pass
        try:
            m[constraints[0].__dict__['a']] = ~constraints[0].__dict__['mask']
        except Exception:
            pass
        try:
            m[constraints[-1].__dict__['a']] = ~(constraints[-1].__dict__['mask'])
        except Exception:
            pass
    index_mask_constraints = np.argwhere(m.reshape(-1))
    return index_mask_constraints


@parallel_function
def apply_mask(list_to_mask=None, index_mask=None):
    """
    Applies mask generated in 'create_mask' in order to hide the coordinates
    that will have no correlation in the GP.
    """
    masked_list = np.zeros((len(list_to_mask), len(index_mask)))
    for i in range(0, len(list_to_mask)):
        masked_list[i] = list_to_mask[i][index_mask].flatten()
    return masked_list


class GPCalculator(Calculator):
    """
    Gaussian Process calculator.
    """
    implemented_properties = ['energy', 'forces', 'uncertainty']
    nolabel = True

    def __init__(self, parameters, get_variance=False, **kwargs):

        Calculator.__init__(self, **kwargs)
        self.gp = parameters
        self.get_variance = get_variance

    def calculate(self, atoms=None,
                  properties=['energy', 'forces', 'uncertainty'],
                  system_changes=all_changes):

        # Atoms object.
        self.atoms = atoms
        Calculator.calculate(self, atoms, properties, system_changes)

        x = self.atoms.get_positions().reshape(-1)

        # Mask geometry to be compatible with the trained GP.
        if self.gp.index_mask is not None:
            index_mask = self.gp.index_mask
            x = apply_mask([x], index_mask=index_mask)[0]

        # Get predictions.
        X = self.gp.X
        kernel = self.gp.kernel
        prior = self.gp.prior
        a = self.gp.a

        n = X.shape[0]
        k = kernel.kernel_vector(x, X, n)
        f = prior.prior(x) + np.dot(k, a)

        energy = f[0]

        forces = -f[1:].reshape(-1)
        forces_empty = np.zeros_like(self.atoms.get_positions().flatten())
        for i in range(len(index_mask)):
            forces_empty[index_mask[i]] = forces[i]
        forces = forces_empty.reshape(-1, 3)

        uncertainty = 0.0

        if self.get_variance:
            v = k.T.copy()
            L = self.gp.L
            v = solve_triangular(L, v, lower=True, check_finite=False)
            variance = kernel.kernel(x, x)
            covariance = np.tensordot(v, v, axes=(0, 0))
            V = variance - covariance
            uncertainty = np.sqrt(V[0][0])
            uncertainty -= self.gp.noise
            if uncertainty < 0.0:
                uncertainty = 0.0

        # Results:
        self.results['energy'] = energy
        self.results['forces'] = forces
        self.results['uncertainty'] = uncertainty
