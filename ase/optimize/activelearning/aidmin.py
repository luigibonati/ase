from ase.optimize.optimize import Optimizer
from ase.optimize.bfgslinesearch import BFGSLineSearch

from ase.optimize.activelearning.gp.calculator import GPCalculator
from ase.optimize.activelearning.trainingset import TrainingSet

from ase.calculators.calculator import compare_atoms

import warnings
from scipy.optimize import minimize


__all__ = ['AIDMin']


class AIDMin(Optimizer):

    def __init__(self, atoms, restart=None, logfile='-', trajectory=None,
                 master=None, force_consistent=None, model_calculator=None,
                 optimizer=BFGSLineSearch, use_previous_observations=False,
                 surrogate_starting_point='min', trainingset=None,
                 print_format='ASE', fit_to='calc', optimizer_kwargs=None,
                 low_memory=False):
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

        self.low_memory = low_memory

        # Initialize the optimizer class
        Optimizer.__init__(self, atoms, restart, logfile, trajectory,
                           master, force_consistent)

        # Model calculator:
        self.model_calculator = model_calculator
        # Default GP Calculator parameters if not specified by the user
        if model_calculator is None:
            self.model_calculator = GPCalculator(
                train_images=[],
                params={'scale': 0.3, 'weight': 2.},
                noise=0.003, update_prior_strategy='fit',
                max_train_data_strategy='nearest_observations',
                max_train_data=5, calculate_uncertainty=False)

        if hasattr(self.model_calculator, 'print_format'):
            self.model_calculator.print_format = print_format

        # Active Learning setup (single-point calculations).
        self.function_calls = 0
        self.force_calls = 0

        self.optimizer = optimizer
        if optimizer_kwargs is not None:
            self.optkwargs = optimizer_kwargs.copy()
        else:
            self.optkwargs = {}
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
                         use_previous_observations=use_previous_observations)

        # Make the training set an observer for the run
        self.attach(self.train, atoms=atoms, method='min')

        # Default parameters for the surrogate optimizer
        if 'logfile' not in self.optkwargs.keys():
            self.optkwargs['logfile'] = None
        if 'trajectory' not in self.optkwargs.keys():
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
        self.ml_fmax = self.optkwargs.get('fmax', None)

        # Now correct it with user define choice in run
        if ml_fmax is not None:
            self.ml_fmax = ml_fmax

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

        # Remove constraints.
        for img in train_images:
            if self.fit_to == 'calc':
                img.constraints = []
            elif self.fit_to == 'constraints':
                img.set_constraint(self.constraints)

        # 2. Update model calculator.
        self.model_calculator.update_train_data(train_images=train_images,
            test_images=[self.atoms.copy()])

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

        # kwargs does not know fmax
        kwargs = self.optkwargs.copy()
        if 'fmax' in kwargs:
            del kwargs['fmax']

        def optimize():
            # Optimize
            opt = self.optimizer(ml_atoms, **kwargs)
            ml_converged = opt.run(fmax=self.ml_fmax, steps=self.ml_steps)

            if not ml_converged:
                raise RuntimeError(
                    "The minimization of the surrogate PES has not converged")
            else:
                # Check that this was a meaningfull step
                system_changes = compare_atoms(atoms, ml_atoms)

                if not system_changes:
                    raise RuntimeError("Too small step: the atoms did not move")

        # Optimize
        optimize()
        
        # Initialize convergence flag for including points in data set
        data_converged = False if self.low_memory else True
        probed_atoms = [atoms.copy()]
        while not data_converged:
            probed_atoms.append(ml_atoms.copy())
            ml_atoms._calc.update_train_data(train_images=[],
                test_images=probed_atoms)
            optimize()
            p1 = probed_atoms[-1].get_positions()
            p0 = ml_atoms.get_positions()
            dist = ((p1-p0)*(p1-p0)).sum(axis=1).max()
            if dist<=1e-4**2:
                break

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


class MaxedOut(Exception):
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

        self.fmax = fmax
        # Set convergence criterium
        if self.fmax == 'scipy default':
            # Scipys default convergence
            tol = None
        else:
            # ASE's usual behaviour
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
        except MaxedOut:
            return False
        else:
            if result.success is False:
                raise RuntimeError('SciPy Error: ' + str(result.message))
            elif tol is None:
                # Scipy's default convergence was met
                return True

            return False

    def step(self):
        pass

    def converged(self, forces=None):
        if self.fmax == 'scipy default':
            return False
        else:
            return Optimizer.converged(forces)

    def func(self, x):
        self.atoms.set_positions(x.reshape((-1, 3)))
        f = self.atoms.get_forces()
        self.log(f)
        self.call_observers()
        if self.converged(f):
            raise Converged
        self.nsteps += 1
        if self.nsteps == self.max_steps:
            raise MaxedOut

        e = self.atoms.get_potential_energy(
            force_consistent=self.force_consistent)
        return e, -f.ravel()


class GPMin(AIDMin):
    def __init__(self, atoms, restart=None, logfile='-', trajectory=None,
                 prior=None, kernel=None, master=None, noise=None, weight=None,
                 scale=None, force_consistent=None, batch_size=None,
                 bounds=None, update_prior_strategy='maximum',
                 update_hyperparams=False):

               """Optimize atomic positions using GPMin algorithm, which uses both
        potential energies and forces information to build a PES via Gaussian
        Process (GP) regression and then minimizes it.

        Default behaviour:
        --------------------
        The default values of the scale, noise, weight, batch_size and bounds
        parameters depend on the value of update_hyperparams. In order to get
        the default value of any of them, they should be set up to None.
        Default values are:

        update_hyperparams = True
            scale : 0.3
            noise : 0.004
            weight: 2.
            bounds: 0.1
            batch_size: 1

        update_hyperparams = False
            scale : 0.4
            noise : 0.005
            weight: 1.
            bounds: irrelevant
            batch_size: irrelevant

        Parameters:
        ------------------

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

        noise: float
            Regularization parameter for the Gaussian Process Regression.

        weight: float
            Prefactor of the Squared Exponential kernel.
            If *update_hyperparams* is False, changing this parameter
            has no effect on the dynamics of the algorithm.

        update_prior_strategy: string
            Strategy to update the constant from the ConstantPrior
            when more data is collected. It does only work when
            Prior = None

            options:
                'maximum': update the prior to the maximum sampled energy
                'init' : fix the prior to the initial energy
                'average': use the average of sampled energies as prior

        scale: float
            scale of the Squared Exponential Kernel

        update_hyperparams: boolean
            Update the scale of the Squared exponential kernel
            every batch_size-th iteration by maximizing the
            marginal likelihood.

        batch_size: int
            Number of new points in the sample before updating
            the hyperparameters.
            Only relevant if the optimizer is executed in update_hyperparams
            mode: (update_hyperparams = True)

        bounds: float, 0<bounds<1
            Set bounds to the optimization of the hyperparameters.
            Let t be a hyperparameter. Then it is optimized under the
            constraint (1-bound)*t_0 <= t <= (1+bound)*t_0
            where t_0 is the value of the hyperparameter in the previous
            step.
            If bounds is False, no constraints are set in the optimization of
            the hyperparameters.

        .. warning:: The memory of the optimizer scales as O(n²N²) where
                     N is the number of atoms and n the number of steps.
                     If the number of atoms is sufficiently high, this
                     may cause a memory issue.
                     This class prints a warning if the user tries to
                     run GPMin with more than 100 atoms in the unit cell.
        """



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

            params_to_update = {'weight': bounds, 'scale': bounds}

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
            params_to_update = []

        # 3. Set GP calculator
        gp_calc = GPCalculator(train_images=None, noise=noise,
                               params={'weight': weight,
                                       'scale': scale},
                               update_prior_strategy=update_prior_strategy,
                               calculate_uncertainty=False,
                               prior=prior, kernel=kernel,
                               params_to_update=params_to_update,
                               batch_size=batch_size,
                               mask_constraints=False)

        # 4. Initialize AIDMin under this set of parameters
        AIDMin.__init__(self, atoms, restart=restart, logfile=logfile,
                        trajectory=trajectory, master=master,
                        force_consistent=force_consistent,
                        model_calculator=gp_calc, optimizer=SP,
                        use_previous_observations=False,
                        surrogate_starting_point='min',
                        trainingset=[], print_format='ASE',
                        fit_to='constraints',
                        optimizer_kwargs={'fmax': 'scipy default',
                                          'method': 'L-BFGS-B'})




#------------------------------
#    BondMin
#------------------------------

class BondMin(AIDMin):
    """
    BondMin optimizer. 
    Optimize atomic positions using GPMin algorithm, which uses both
    potential energies and forces information to build a PES via Gaussian
    Process (GP) regression and then minimizes it.

    Default behaviour:
    --------------------
    The default values of the scale, noise, weight, batch_size and bounds
    parameters depend on the value of update_hyperparams. In order to get
    the default value of any of them, they should be set up to None.
    Default values are:

    update_hyperparams = True
        scale : 0.2
        noise : 0.01
        weight: 2.
        bounds: 0.1
        batch_size: 1

    update_hyperparams = False
        scale : 0.4
        noise : 0.005
        weight: 1.
        bounds: irrelevant
        batch_size: irrelevant


    By default, the full memory version is used. The light memory version can 
    be enabled by setting max_data to an integer. This functionality reduces 
    the number of point in the training set to those who are closest to 
    the test point and does not fit to the forces of constrained atoms.   


    Parameters:
    ------------------

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

    noise: float
        Regularization parameter for the Gaussian Process Regression.

    weight: float
        Prefactor of the Squared Exponential kernel.
        If *update_hyperparams* is False, changing this parameter
        has no effect on the dynamics of the algorithm.

    update_prior_strategy: string
        Strategy to update the constant from the ConstantPrior
        when more data is collected. It does only work when
        Prior = None

        options:
            'maximum': update the prior to the maximum sampled energy
            'init' : fix the prior to the initial energy
            'average': use the average of sampled energies as prior

    scale: float
        global scale of the method

    update_hyperparams: boolean
        Update hyperparameters of the kernel.

    batch_size: int
        Number of new points in the sample before updating
        the hyperparameters.
        Only relevant if the optimizer is executed in update_hyperparams
        mode: (update_hyperparams = True)

    bounds: float, 0<bounds<1
        Set bounds to the optimization of the hyperparameters.
        Let t be a hyperparameter. Then it is optimized under the
        constraint (1-bound)*t_0 <= t <= (1+bound)*t_0
        where t_0 is the value of the hyperparameter in the previous
        step.
        If bounds is False, no constraints are set in the optimization of
        the hyperparameters.

    max_train_data (default: None)
        maximum number of points in the training set for light memory.
        If None, it is infinetly many points.

    .. warning:: The memory of the optimizer scales as O(n²N²) where
                 N is the number of atoms and n is max_data.
                 If the number of atoms is sufficiently high, this
                 may cause a memory issue for a given max_data.
                 If the number of points in the training set is not
                 restricted, this class prints a warning if the user tries to
                 run BonfMin with more than 100 atoms in the unit cell.
    """

    def __init__(self, atoms, restart=None, logfile='-', trajectory=None,
                 prior=None,
                 master=None, noise=None, weight=None,
                 scale=None, force_consistent=None, batch_size=None,
                 bounds=None, update_prior_strategy=None,
                 update_hyperparams=True,
                 max_train_data=None):

     
        # 1. Warn the user if the number of atoms is very large
        if max_train_data is None and len(atoms) > 100:
            warning = ('Possible Memeroy Issue. There are more than '
                       '100 atoms in the unit cell. The memory '
                       'of the process will increase with the number '
                       'of steps, potentially causing a memory issue. '
                       'Consider using a different optimizer.')

            warnings.warn(warning)

        # 2. Define interaction and kernel
        def inv_sqsum(A, B):
            rA = covalent_radii[atomic_numbers[A]]
            rB = covalent_radii[atomic_numbers[B]]
            return 4 * (rA + rB)**(-2)

        kernel = BondExponential()
        symbols = atoms.get_chemical_symbols()
        kernel.init_metric(symbols, inv_sqsum)


        # 3. Define default hyperparameters
        #  3.A Updated BondedMin
        if update_hyperparams:
            if scale is None:
                scale = 0.2
            if noise is None:
                noise = 0.01
            if weight is None:
                weight = 2.

            if bounds is None:
                bounds = 0.1
            elif bounds is False:
                bounds = None

            if batch_size is None:
                batch_size = 1

            if update_prior_strategy is None:
                update_prior_strategy = 'fit'
            fit_weight = True

            # Add the weight and the bond hyperparameters to update
            params_to_update = {'weight': bounds}
            for param in kernel.params.keys():
                if param.startswith('l_'):
                    params_to_update[param] = bounds

        #  3.B Un-updated BondedMin
        else:
            if scale is None:
                scale = 0.4
            if noise is None:
                noise = 0.0025
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
            params_to_update = []

            fit_weight = False

            if update_prior_strategy is None:
                update_prior_strategy = 'maximum'


        # 4. Light memory version
        if max_train_data is not None:
            mask_constraints = True
        else:
            mask_constraints = False


        # 5. Set GP calculator
        gp_calc = GPCalculator(train_images=None, noise=noise,
                               params={'weight': weight,
                                       'scale': scale},
                               update_prior_strategy=update_prior_strategy,
                               calculate_uncertainty=False,
                               prior=prior, kernel=kernel,
                               params_to_update=params_to_update,
                               batch_size=batch_size,fit_weight=fit_weight,
                               max_train_data=max_train_data,
                               max_data_strategy='nearest_observations',
                               mask_constraints=mask_constraints)

        # 6. Initialize AIDMin under this set of parameters
        AIDMin.__init__(self, atoms, restart=restart, logfile=logfile,
                        trajectory=trajectory, master=master,
                        force_consistent=force_consistent,
                        model_calculator=gp_calc, optimizer=SP,
                        use_previous_observations=False,
                        surrogate_starting_point='min',
                        trainingset=[], print_format='ASE',
                        fit_to='calc',
                        optimizer_kwargs={'fmax': 'scipy default',
                                          'method': 'L-BFGS-B'})
