from ase.calculators.gp.kernel import SquaredExponential
from ase.calculators.gp.gp import GaussianProcess
from ase.calculators.gp.prior import ConstantPrior
from ase.calculators.calculator import Calculator, all_changes
from scipy.linalg import solve_triangular
import numpy as np
import time


class GPCalculator(Calculator, GaussianProcess):
    """
    GP model parameters
    -------------------
    prior: Prior object or None
        Prior for the GP regression of the PES surface. See
        ase.optimize.activelearning.prior. If *Prior* is None, then it is set
        as the ConstantPrior with the constant being updated using the
        update_prior_strategy specified as a parameter.

    weight: float
        Pre-exponential factor of the Squared Exponential kernel. If
        *update_hyperparams* is False, changing this parameter has no effect
        on the dynamics of the algorithm.

    scale: float
        Scale of the Squared Exponential Kernel.

    noise: float
        Regularization parameter for the Gaussian Process Regression.

    update_prior_strategy: string
        Strategy to update the constant from the ConstantPrior when more
        data is collected. It does only work when Prior = None

        options:
            'maximum': update the prior to the maximum sampled energy.
            'average': use the average of sampled energies as prior.
            'init' : fix the prior to the initial energy.
            'last' : fix the prior to the last sampled energy.

    update_hyperparams: boolean
        Update the scale of the Squared exponential kernel every
        batch_size-th iteration by maximizing the marginal likelihood.

    batch_size: int
        Number of new points in the sample before updating the hyperparameters.
        Only relevant if the optimizer is executed in update
        mode: (update = True)

    bounds: float, 0<bounds<1
        Set bounds to the optimization of the hyperparameters. Let t be a
        hyperparameter. Then it is optimized under the constraint (
        1-bound)*t_0 <= t <= (1+bound)*t_0 where t_0 is the value of the
        hyperparameter in the previous step. If bounds is None,
        no constraints are set in the optimization of the hyperparameters.

    max_train_data: int
        Number of observations that will effectively be included in the GP
        model. See also *max_data_strategy*.

    max_train_data_strategy: string
        Strategy to decide the observations that will be included in the model.

        options:
            'last_observations': selects the last observations collected by
            the surrogate.
            'lowest_energy': selects the lowest energy observations
            collected by the surrogate.

        For instance, if *max_train_data* is set to 50 and
        *max_train_data_strategy* to 'lowest energy', the surrogate model
        will be built in each iteration with the 50 lowest energy
        observations collected so far.

    """

    implemented_properties = ['energy', 'forces', 'uncertainty']
    nolabel = True

    def __init__(self, train_images=None, prior=None,
                 update_prior_strategy='maximum', weight=1.0,
                 scale=0.4, noise=0.005, update_hyperparams=False,
                 batch_size=5, bounds=None, kernel=None,
                 max_train_data=None, force_consistent=None,
                 max_train_data_strategy='last_observations', **kwargs):

        Calculator.__init__(self, **kwargs)
        self.prior = prior
        self.strategy = update_prior_strategy
        self.weight = weight
        self.scale = scale
        self.noise = noise
        self.update_hp = update_hyperparams
        self.nbatch = batch_size
        self.hyperbounds = bounds
        self.force_consistent = force_consistent
        self.max_data = max_train_data
        self.max_data_strategy = max_train_data_strategy
        self.kernel = kernel
        self.train_images = train_images
        self.old_train_images = []
        self.calculate_uncertainty = True

    def initialize(self):
        """ Initialize the calculator, including model parameters. """

        self.train_x = []
        self.train_y = []

        # Set kernel and prior.
        if self.kernel is None:
            self.kernel = SquaredExponential()

        if self.prior is None:
            self.update_prior = True
            self.prior = ConstantPrior(constant=None)

        else:
            self.update_prior = False

        # Set Gaussian Process parameters.
        GaussianProcess.__init__(self, self.prior, self.kernel)

        # Set initial hyperparameters.
        self.set_hyperparams(np.array([self.weight, self.scale, self.noise]))

        # Masks the coordinates of the atoms that are kept fixed (memory).
        self.atoms_mask = self.create_mask()

    def extract_features(self):
        """ From the training images (which include the observations),
        collect the positions, energies and forces required to train the
        Gaussian Process. """

        start = time.time()
        for i in self.train_images:
            r = i.get_positions().reshape(-1)
            e = i.get_potential_energy(force_consistent=self.force_consistent)
            f = i.get_forces()
            self.train_x.append(r[self.atoms_mask])
            y = np.append(np.array(e).reshape(-1),
                          -f.reshape(-1)[self.atoms_mask])
            self.train_y.append(y)
        # self.train_x = np.unique(self.train_x, axis=0)  # Remove duplicates.
        # self.train_y = np.unique(self.train_x, axis=0)  # Remove duplicates.
        end = time.time()
        print('Elapsed time featurizing data:', end-start)

    def update_train_data(self, train_images):
        """ Update the model with observations (feeding new training images),
        after instantiating the GPCalculator class."""

        self.train_images = self.old_train_images
        for i in train_images:
            if i not in self.train_images:
                self.train_images += train_images
        self.calculate(atoms=self.train_images[0])  # Test one to attach.

    def train_model(self):
        """ Train a model with the previously fed observations."""

        # 1. Set/update the the prior.
        if self.update_prior:
            if self.strategy == 'average':
                av_e = np.mean(np.array(self.train_y)[:, 0])
                self.prior.set_constant(av_e)
            elif self.strategy == 'maximum':
                max_e = np.max(np.array(self.train_y)[:, 0])
                self.prior.set_constant(max_e)
            elif self.strategy == 'init':
                self.prior.set_constant(np.array(self.train_y)[:, 0][0])
                self.update_prior = False
            elif self.strategy == 'last':
                self.prior.set_constant(np.array(self.train_y)[:, 0][-1])
                self.update_prior = False

        # 2. Max number of observations consider for training (low memory).
        if self.max_data is not None:
            # 2.a. Get only the last observations.
            if self.max_data_strategy == 'last_observations':
                self.train_x = self.train_x[-self.max_data:]
                self.train_y = self.train_y[-self.max_data:]

            # 2.b. Get the minimum energy observations.
            if self.max_data_strategy == 'lowest_energy':
                e_list = []
                for i in self.train_y:
                    e_list.append(i[0])
                arg_low_e = np.argsort(e_list)[:self.max_data]
                x = [self.train_x[i] for i in arg_low_e]
                y = [self.train_y[i] for i in arg_low_e]
                self.train_x = x
                self.train_y = y

        # 3. Train a Gaussian Process.
        start = time.time()
        self.train(np.array(self.train_x), np.array(self.train_y),
                   noise=self.noise)
        end = time.time()
        print('Elapsed time training process:', end-start)

        # 4. (optional) Optimize model hyperparameters.
        if self.update_hp and len(self.train_x) % self.nbatch == 0 and len(self.train_x) != 0:
            ratio = self.noise / self.kernel.weight
            try:
                self.fit_hyperparameters(np.asarray(self.train_x),
                                         np.asarray(self.train_y),
                                         eps=self.hyperbounds)
            except Exception:
                pass

            else:
                # Keeps the ratio between noise and weight fixed.
                self.noise = ratio * self.kernel.weight

    def calculate(self, atoms=None,
                  properties=['energy', 'forces', 'uncertainty'],
                  system_changes=all_changes):
        """ Calculate the energy, forces and uncertainty on the energies for a
        given Atoms structure. Predicted energies can be obtained by
        *atoms.get_potential_energy()*, predicted forces using
        *atoms.get_forces()* and uncertainties using
        *atoms.get_calculator().results['uncertainty'].
        """

        # Execute training process when *calculate* is called.
        if self.train_images is not None:
            self.initialize()
            self.extract_features()
            self.train_model()
            self.old_train_images = self.train_images.copy()
            self.train_images = None  # Remove the training list of images.

        # Atoms object.
        self.atoms = atoms
        Calculator.calculate(self, atoms, properties, system_changes)

        # Mask geometry to be compatible with the trained GP (reduce memory).
        x = self.atoms.get_positions().reshape(-1)[self.atoms_mask]

        # Get predictions.
        n = self.X.shape[0]
        k = self.kernel.kernel_vector(x, self.X, n)
        f = self.prior.prior(x) + np.dot(k, self.a)

        # Obtain energy and forces for the given geometry.
        energy = f[0]
        forces = -f[1:].reshape(-1)
        forces_empty = np.zeros_like(self.atoms.get_positions().flatten())
        for i in range(len(self.atoms_mask)):
            forces_empty[self.atoms_mask[i]] = forces[i]
        forces = forces_empty.reshape(-1, 3)

        # Get uncertainty for the given geometry.
        uncertainty = 0.0
        if self.calculate_uncertainty is True:
            x = self.atoms.get_positions().reshape(-1)[self.atoms_mask]
            n = self.X.shape[0]
            k = self.kernel.kernel_vector(x, self.X, n)
            v = k.T.copy()
            v = solve_triangular(self.L, v, lower=True, check_finite=False)
            variance = self.kernel.kernel(x, x)
            covariance = np.tensordot(v, v, axes=(0, 0))
            V = variance - covariance
            uncertainty = np.sqrt(V[0][0])
            uncertainty -= self.noise
            if uncertainty < 0.0:
                uncertainty = 0.0

        # Results:
        self.results['energy'] = energy
        self.results['forces'] = forces
        self.results['uncertainty'] = uncertainty

    def create_mask(self):
        """
        This function mask atoms coordinates that will not participate in the
        model, i.e. the coordinates of the atoms that are kept fixed
        or constraint.
        """
        atoms = self.train_images[0]
        constraints = atoms.constraints
        mask_constraints = np.ones_like(atoms.positions, dtype=bool)
        for i in range(0, len(constraints)):
            try:
                mask_constraints[constraints[i].a] = ~constraints[i].mask
            except Exception:
                pass

            try:
                mask_constraints[constraints[i].index] = False
            except Exception:
                pass

            try:
                mask_constraints[constraints[0].a] = ~constraints[0].mask
            except Exception:
                pass

            try:
                mask_constraints[constraints[-1].a] = ~constraints[-1].mask
            except Exception:
                pass
        return np.argwhere(mask_constraints.reshape(-1)).reshape(-1)

