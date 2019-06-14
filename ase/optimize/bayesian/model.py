from scipy.linalg import solve_triangular
from ase.optimize.gpmin.kernel import SquaredExponential
from ase.optimize.gpmin.gp import GaussianProcess
from ase.optimize.gpmin.prior import ConstantPrior
from ase.calculators.calculator import Calculator, all_changes
from ase.parallel import parallel_function, parprint
import numpy as np

class GPModel(GaussianProcess):
    """ GP model parameters
        -------------------
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
            optimization of the hyperparameters."""

    def __init__(self, prior=None, update_prior_strategy='maximum', weight=1.0, scale=0.35,
                 noise=0.005, update_hyperparams=False, batch_size=5, bounds=None, kernel=None,
                 max_training_data=None):
        self.prior = prior
        self.strategy = update_prior_strategy
        self.weight = weight
        self.scale = scale
        self.noise = noise
        self.update_hp = update_hyperparams
        self.nbatch = batch_size
        self.eps = bounds

        # Initialize:
        self.train_x = []
        self.train_y = []
        self.pred_energy = []
        self.pred_uncertainty = []
        self.pred_positions = []
        self.pred_fmax = []
        self.force_consistent = None
        self.max_training_data = max_training_data
        self.atoms_mask = None

        # Set kernel and prior.
        if kernel is None:
            kernel = SquaredExponential()

        if prior is None:
            self.update_prior = True
            prior = ConstantPrior(constant=None)

        else:
            self.update_prior = False

        # Set kernel and prior.
        GaussianProcess.__init__(self, prior, kernel)

        # Set initial hyperparameters.
        self.set_hyperparams(np.array([weight, scale, noise]))

    @parallel_function
    def extract_features(self, train_atoms, force_consistent=None):
        self.force_consistent = force_consistent  # We need to keep it for the test data.
        positions = []
        gradients = []
        targets = []
        for i in train_atoms:
            positions.append(i.get_positions().reshape(-1))
            targets.append(i.get_potential_energy(force_consistent=self.force_consistent))
            gradients.append(-i.get_forces().reshape(-1))

        positions = np.reshape(positions, (len(positions), -1))
        gradients = np.reshape(gradients,  (len(gradients), -1))
        targets = np.reshape(targets, (len(targets), -1))

        # Create mask to hide atoms that do not participate in the model (constraints):
        self.atoms_mask = create_mask(train_atoms[0])

        if self.atoms_mask is not None:
            positions = apply_mask(list_to_mask=positions, mask_index=self.atoms_mask)
            gradients = apply_mask(list_to_mask=gradients, mask_index=self.atoms_mask)

        self.train_x = positions
        self.train_y = np.block([targets, gradients])

        # Set/update the constant for the prior.
        if self.update_prior:
            if self.strategy == 'average':
                av_e = np.mean(np.array(targets)[:, 0])
                self.prior.set_constant(av_e)
            elif self.strategy == 'maximum':
                max_e = np.max(np.array(targets)[:, 0])
                self.prior.set_constant(max_e)
            elif self.strategy == 'init':
                self.prior.set_constant(targets[0])
                self.update_prior = False

    @parallel_function
    def train_model(self):
        train_x = np.array(self.train_x)
        train_y = np.array(self.train_y)

        # Train the model.
        self.train(self.train_x, self.train_y, noise=self.noise)

        # Optimize hyperparameters (optional).
        if self.update_hp and len(train_x) % self.nbatch == 0 and len(train_x) != 0:
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

    @parallel_function
    def get_atoms_predictions(self, test_atoms, get_uncertainty=True):
        """
        Obtain predictions from the model.
        """
        list_pred_u = []
        list_pred_e = []
        list_pred_r = []
        list_pred_fmax = []
        for i in test_atoms:
            i.set_calculator(GPCalculator(parameters=self, get_variance=get_uncertainty))
            list_pred_e.append(i.get_potential_energy(force_consistent=self.force_consistent))
            list_pred_r.append(i.positions.reshape(-1))
            list_pred_fmax.append(np.sqrt((i.get_forces()**2).sum(axis=1).max()))
            uncertainty = i.get_calculator().results['uncertainty']
            list_pred_u.append(uncertainty)
            i.info['uncertainty'] = uncertainty
        self.pred_energy = list_pred_e
        self.pred_uncertainty = list_pred_u
        self.pred_positions = list_pred_r
        self.pred_fmax = list_pred_fmax


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
        if self.gp.atoms_mask is not None:
            index_mask = self.gp.atoms_mask
            x = apply_mask([x], mask_index=index_mask)[0]

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


def create_mask(atoms):
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
    mask_constraints = mask_constraints.reshape(-1)
    atoms_mask_constraints = np.argwhere(mask_constraints)
    return atoms_mask_constraints


def apply_mask(list_to_mask=None, mask_index=None):
    masked_list = np.zeros((len(list_to_mask), len(mask_index)))
    for i in range(0, len(list_to_mask)):
        masked_list[i] = list_to_mask[i][mask_index].flatten()
    return masked_list
