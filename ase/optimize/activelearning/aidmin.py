from ase.optimize.optimize import Optimizer
from ase.optimize.bfgslinesearch import BFGSLineSearch
from ase.optimize.activelearning.gp.calculator import GPCalculator
from ase.optimize.activelearning.trainingset import TrainingSet

import warnings
from scipy.optimize import minimize


__all__ = ['AIDMin']


class AIDMin(Optimizer):

    def __init__(self, atoms, restart=None, logfile='-', trajectory=None,
                 master=None, force_consistent=None, model_calculator=None,
                 optimizer=BFGSLineSearch, use_previous_observations=False,
                 surrogate_starting_point='min', trainingset=None,
                 print_format='ASE', fit_to='calc', optimizer_kwargs={}):
        """
        Artificial Intelligence-Driven energy Minimizer (AID-Min) algorithm.
        Optimize atomic structure using a surrogate machine learning
        model [1,2]. Atomic positions, potential energies and forces
        information are used to build a modelled potential energy surface (
        PES) that can be optimized to obtain new suggested structures
        towards finding a local minima in the targeted PES.

        [1] E. Garijo del Rio, J. J. Mortensen and K. W. Jacobsen.
        arXiv:1808.08588.
        [2] J. A. Garrido Torres, E. Garijo del Rio, A. H. Larsen,
        V. Streibel, J. J. Mortensen, M. Bajdich, F. Abild-Pedersen,
        K. W. Jacobsen, T. Bligaard. (submitted).

        Parameters
        --------------

        atoms: Atoms object
            The Atoms object to relax.

        restart: string
            Pickle file used to store the training set. If set, file with
            such a name will be searched and the data in the file incorporated
            to the new training set, if the file exists.

        logfile: file object or str
            If *logfile* is a string, a file with that name will be opened.
            Use '-' for stdout

        trajectory: string
            Pickle file used to store trajectory of atomic movement.

        master: boolean
            Defaults to None, which causes only rank 0 to save files. If
            set to True, this rank will save files.

        force_consistent: boolean or None
            Use force-consistent energy calls (as opposed to the energy
            extrapolated to 0 K). By default (force_consistent=None) uses
            force-consistent energies if available in the calculator, but
            falls back to force_consistent=False if not.

        DOCUMENTATION MISSING!!

        use_previous_observations: boolean
            If False. The optimization starts from scratch.
                If the observations were saved to a trajectory file,
                it is overwritten. If they were kept in a list, they are
                deleted.
            If True. The optimization uses the information that was already
                in the training set that is provided in the optimization.

        surrogate_starting_point: string
               Where to start the minimization from.
               options:
                     'last' : start minimization of the surrogate from the
                              last position
                     'min': start minimzation from the position with
                              the lowest energy of those visited. This option
                              is the one chosen in GPMin [1] and is the
                              recommended one to avoid extrapolation.

        trainingset: None, trajectory file or list
            Where the training set is kept, either saved to disk in a
            trajectory file or kept in memory in a list.
            options:
                None (default):
                    A trajectory file named *trajectory*_observations.traj is
                    automatically generated and the training set is saved to
                    it while generated.
                str: trajectory filename where to append the training set
                list: list were to append the atoms objects.

        print_format: string
            Printing format. It chooses how much information and in which
            format is shown to the user.

            options:
                  'ASE' (default): information printed matches other ASE
                      functions outside from the AID module. ML is transparent
                      to the user.
                  'AID': Original format of the AID module. More informative
                      in respect of ML process.
                      This option is advised for experienced users.
        fit_to: string
            Characteristics of the constraints in the training set.

            options:
                'calc' (default): fit to the output of the calculator, then
                    run over the constrained surface.
                'constraints': fit to the constrained atoms directly

        optimizer_kwargs: dict
            Dictionary with key-word arguments for the surrogate potential.
        """
        if print_format == 'AID':
            if logfile == '-':
                logfile = None
            self.print_format = 'AID'
        else:
            self.print_format = 'ASE'

        # Initialize the optimizer class
        Optimizer.__init__(self, atoms, restart, logfile, trajectory,
                           master, force_consistent)

        # Model calculator:
        self.model_calculator = model_calculator
        # Default GP Calculator parameters if not specified by the user
        if model_calculator is None:
            self.model_calculator = GPCalculator(
                        train_images=[], scale=0.3, weight=2.,
                        noise=0.003, update_prior_strategy='fit',
                        max_train_data_strategy='nearest_observations',
                        max_train_data=5,
                        calculate_uncertainty=False)

        if hasattr(self.model_calculator, 'print_format'):
            self.model_calculator.print_format = print_format

        # Active Learning setup (single-point calculations).
        self.function_calls = 0
        self.force_calls = 0

        self.optimizer = optimizer
        self.optkwargs = optimizer_kwargs
        self.start = surrogate_starting_point

        self.constraints = atoms.constraints.copy()

        # Initialize training set
        if trainingset is None:
            trajectory_main = trajectory.split('.')[0]
            self.train = TrainingSet(trajectory_main + '_observations.traj',
                                     use_previous_observations=False)
        elif isinstance(trainingset, TrainingSet):
            self.train = trainingset
        else:
            self.train = TrainingSet(trainingset,
                                     use_previous_observations=
                                     use_previous_observations)

        # Make the training set an observer for the run
        self.attach(self.train, atoms=atoms, method='min')

        # Default parameters for the surrogate optimizer
        if 'logfile' not in self.optkwargs.items():
            self.optkwargs['logfile'] = None
        if 'trajectory' not in self.optkwargs.items():
            self.optkwargs['trajectory'] = None

        # Define what to fit to
        if fit_to not in ('calc', 'constraints'):
            raise ValueError("fit_to must be either 'calc' or 'constraints'.")
        self.fit_to = fit_to

    def set_trainingset(self, trainingset, substitute=True, atoms=None):
        """
        Sets a new TrainingSet object to store the training set
        Parameters:
        -----------
        trainingset: TrainingSet object
            training set to be attached to the Optimizer
        substitute: bool
            whether to remove the previous Training Set from the
            observers list or not
        atoms: Atoms object or None
           Atoms object to be attached. If None, it is the same
           as in the optimizer
        """

        if atoms is None:
            atoms = self.atoms

        self.train = trainingset

        # Remove previous training set
        def isTrainingSet(observer):
            try:
                return observer[0].__qualname__.startswith('TrainingSet')
            except AttributeError:
                return False

        if substitute:
            self.observers[:] = [obs for obs in self.observers
                                 if not isTrainingSet(obs)]

        # Attach new training set
        self.attach(trainingset, atoms=atoms, method='min')

    def run(self, fmax=0.05, steps=None, ml_steps=500, ml_fmax=None):

        """
        Executing run will start the optimization process.
        This method will return when the forces on all individual
        atoms are less than *fmax* or when the number of steps exceeds
        *steps*.

        Parameters
        ----------
        fmax : float
            Convergence criteria (in eV/Angstrom)

        steps: int
            Maximum number of steps

        ml_steps: int
            Maximum number of steps for the surrogate

        Returns
        -------
        True if the optimizer found a structure with
        fmax below the threshold in nsteps or less.
        False otherwise.
        """

        # Machine Learning parameters
        self.ml_steps = ml_steps

        # First set fmax according to __init__method
        self.ml_fmax = self.optkwargs.pop('fmax', None)

        # Now correct it with user define choice in run
        if ml_fmax is not None:
            self.ml_fmax = fmax

        # Finally, especify default option
        if self.ml_fmax is None:
            self.ml_fmax = 0.01 * fmax

        if self.start == 'min':
            self.check()

        Optimizer.run(self, fmax, steps)

    def step(self):

        """
        Inner function over which Optimizer.run iterates
        """

        # 1. Gather observations in every iteration.
        # This serves to use the previous observations (useful for continuing
        # calculations and/or parallel runs).
        train_images = self.train.load_set()

        # Remove constraints. TODO: do this in a more elegant way
        for img in train_images:
            if self.fit_to == 'calc':
                img.constraints = []
            elif self.fit_to == 'constraints':
                img.set_constraint(self.constraints)

        # 2. Update model calculator.
        self.model_calculator.update_train_data(train_images=train_images)

        # 3. Optimize the structure in the predicted PES
        self.min_surrogate(self.atoms)

        # 4. Evaluate the target function
        self.atoms.get_potential_energy(force_consistent=self.force_consistent)
        self.atoms.get_forces()
        self.function_calls = len(train_images) + 1
        self.force_calls = self.function_calls

        # 5. Check if the performance is sensible
        if self.start == 'min':
            self.check()

    def min_surrogate(self, atoms):

        # Setup atoms obejct in surrogate potential
        ml_atoms = atoms.copy()
        ml_atoms.set_calculator(self.model_calculator)
        if self.start == 'min':
            ml_atoms.set_positions(self.p0)

        # Set constraints
        if self.fit_to == 'constraints':
            ml_atoms.set_constraint([])

        # Optimize
        opt = self.optimizer(ml_atoms, **self.optkwargs)
        ml_converged = opt.run(fmax=self.ml_fmax, steps=self.ml_steps)

        if not ml_converged:
            raise RuntimeError(
                "The minimization of the surrogate PES has not converged")
        else:
            atoms.set_positions(ml_atoms.get_positions())

    def check(self):

        if self.nsteps == 0:
            self.e0 = self.atoms.get_potential_energy(
                force_consistent=self.force_consistent)
            self.p0 = self.atoms.get_positions()
            self.count = 0
        else:
            e = self.atoms.get_potential_energy(
                force_consistent=self.force_consistent)
            if e < self.e0:
                self.e0 = e
                self.p0 = self.atoms.get_positions()
                self.count = 0
            elif self.count < 30:
                self.count += 1
            else:
                raise RuntimeError('A descent model could not be built')


class Converged(Exception):
    pass


class SP(Optimizer):
    """
    Scipy optimizer taking atoms object as input
    """
    def __init__(self, atoms, method='L-BFGS-B', logfile=None,
                 trajectory=None, master=None, force_consistent=None):
        Optimizer.__init__(self, atoms, restart=None,
                           logfile=logfile, trajectory=trajectory,
                           master=master,
                           force_consistent=force_consistent)
        self.method = method

    def run(self, fmax=0.05, steps=100000000):

        # Set convergence criterium
        if fmax == 'scipy default':
            # Scipys default convergence
            self.fmax = 1e-8
            tol = None
        else:
            # ASE's usual behaviour
            self.fmax = fmax
            tol = 1e-10

        self.max_steps = steps

        f = self.atoms.get_forces()

        self.log(f)
        self.call_observers()
        if self.converged(f):
            return True
        try:
            result = minimize(self.func,
                              self.atoms.positions.ravel(),
                              jac=True,
                              method=self.method,
                              tol=tol)
        except Converged:
            return True
        else:
            if result.success is False:
                raise RuntimeError('SciPy Error: ' + result.message)
            elif tol is None:
                # Scipy's default convergence was met
                return True

            return False

    def step(self):
        pass

    def func(self, x):
        self.atoms.set_positions(x.reshape((-1, 3)))
        f = self.atoms.get_forces()
        self.log(f)
        self.call_observers()
        if self.converged(f):
            raise Converged
        self.nsteps += 1
        e = self.atoms.get_potential_energy(
            force_consistent=self.force_consistent)
        return e, -f.ravel()


class GPMin(AIDMin):
    def __init__(self, atoms, restart=None, logfile='-', trajectory=None,
                 prior=None, kernel=None, master=None, noise=None, weight=None,
                 scale=None, force_consistent=None, batch_size=None,
                 bounds=None, update_prior_strategy='maximum',
                 update_hyperparams=False):

        # 1. Warn the user if the number of atoms is very large
        if len(atoms) > 100:
            warning = ('Possible Memeroy Issue. There are more than '
                       '100 atoms in the unit cell. The memory '
                       'of the process will increase with the number '
                       'of steps, potentially causing a memory issue. '
                       'Consider using a different optimizer.')

            warnings.warn(warning)

        # 2. Define default hyperparameters
        #  2.A Updated GPMin
        if update_hyperparams:
            if scale is None:
                scale = 0.3
            if noise is None:
                noise = 0.004
            if weight is None:
                weight = 2.

            if bounds is None:
                bounds = 0.1
            elif bounds is False:
                bounds = None

            if batch_size is None:
                batch_size = 1

        #  2.B Un-updated GPMin
        else:
            if scale is None:
                scale = 0.4
            if noise is None:
                noise = 0.001
            if weight is None:
                weight = 1.

            if bounds is not None:
                warning = ('The paramter bounds is of no use '
                           'if update_hyperparams is False. '
                           'The value provided by the user '
                           'is being ignored.')
                warnings.warn(warning, UserWarning)
            if batch_size is not None:
                warning = ('The paramter batch_size is of no use '
                           'if update_hyperparams is False. '
                           'The value provived by the user '
                           'is being ignored.')
                warnings.warn(warning, UserWarning)

            # Set batch_size to 1 anyways
            batch_size = 1

        # 3. Set GP calculator
        gp_calc = GPCalculator(train_images=[], scale=scale,
                               weight=weight, noise=noise,
                               update_prior_strategy=update_prior_strategy,
                               calculate_uncertainty=False,
                               prior=prior, kernel=kernel,
                               update_hyperparams=update_hyperparams,
                               bounds=bounds, batch_size=batch_size,
                               mask_constraints=False)
        """
        Thoughts:
            1. I have not specified max_train_data_strategy and max_train_data
             to the calculator
        """

        # 4. Initialize AIDMin under this set of parameters
        AIDMin.__init__(self, atoms, restart=restart, logfile=logfile,
                        trajectory=trajectory, master=master,
                        force_consistent=force_consistent,
                        model_calculator=gp_calc, optimizer=SP,
                        use_previous_observations=False,
                        surrogate_starting_point='min',
                        trainingset=[], print_format='ASE',
                        fit_to='constraints',
                        optimizer_kwargs={'fmax': 'scipy default', 'method': 'L-BFGS-B'})

        """
        Thoughts:
              1. Check use_previous_observations still works
              2. max_train_data and max_train_data_strategy
        """
