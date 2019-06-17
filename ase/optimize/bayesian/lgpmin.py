import copy
import time
from ase.parallel import parprint
from ase import io
from ase.atoms import Atoms
from ase.optimize import LBFGS
from ase.optimize.bayesian.io import print_cite_min, dump_experiences, get_fmax
from ase.optimize.bayesian.model import GPModel, create_mask


class LGPMin:

    def __init__(self, atoms, model=None, force_consistent=None):

        """ Optimize atomic structure using a surrogate machine learning
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

        model: Model object.
            Predictive model to be used to build a PES for the surrogate.
            The default is None which uses a GP model with the Squared
            Exponential Kernel and other default parameters. See
            *ase.optimize.bayesian.model* GPModel for default GP parameters.

        force_consistent: boolean or None
            Use force-consistent energy calls (as opposed to the energy
            extrapolated to 0 K). By default (force_consistent=None) uses
            force-consistent energies if available in the calculator, but
            falls back on force_consistent=False if not.

        """
        # Predictive model parameters:
        self.model = model
        if model is None:
            self.model = GPModel()

        # General setup.
        atoms.get_potential_energy(force_consistent=force_consistent)
        self.ase_calc = atoms.get_calculator()
        self.constraints = atoms.constraints

        # Optimization.
        self.function_calls = 0
        self.force_calls = 0
        self.force_consistent = force_consistent

        self.images_pool = [copy.deepcopy(atoms)]

        # Create mask to hide atoms that do not participate in the model:
        if self.constraints is not None:
            self.atoms_mask = create_mask(atoms)

    def run(self, fmax=0.05, ml_steps=200, trajectory='LGPMin.traj',
            restart=False):

        """
        Executing run will start the optimization process.

        Parameters
        ----------
        fmax : float
            Convergence criteria (in eV/Angstrom).

        ml_steps: int
            Maximum number of steps for the Atoms optimization on the GP
            predicted potential energy surface.

        trajectory: string
            Filename to store the optimization of the predicted PES.
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
        Optimized Atoms structure.

        """

        while get_fmax(self.images_pool[-1]) > fmax:

            # 1. Experiences. Restart from a previous (and/or parallel) runs.
            start = time.time()
            dump_experiences(images=self.images_pool,
                             filename=trajectory, restart=restart)
            if restart is True:
                self.images_pool = io.read(trajectory.split('.')[0] +
                                           '_experiences.traj', ':')
            end = time.time()
            parprint('Elapsed time dumping/reading images to build a model:',
                     end-start)

            # 2. Build a ML model from Atoms images (i.e. images_pool).
            start = time.time()
            self.model.extract_features(train_images=self.images_pool,
                                        atoms_mask=self.atoms_mask,
                                        force_consistent=self.force_consistent)
            end = time.time()
            parprint('Elapsed time featurizing data:', end-start)

            # 3. Train the model.
            start = time.time()
            self.model.train_model()
            end = time.time()
            parprint('Elapsed time training the model:', end-start)

            # Attach ML calculator to the Atoms to be optimize.
            test_atoms = copy.deepcopy(self.images_pool[-1])
            self.model.get_images_predictions([test_atoms],
                                              get_uncertainty=True)
            print(test_atoms.get_stresses())
            exit()

            # 4. Optimize predicted PES.
            opt = LBFGS(test_atoms, trajectory=trajectory, logfile=None)
            start = time.time()
            opt.run(fmax=fmax*0.5, steps=ml_steps)
            end = time.time()
            parprint('Elapsed time optimizing predicted PES:', end-start)

            # 5. Acquisition function. Greedy last optimized structure.
            positions_to_evaluate = test_atoms.positions

            # 6. Evaluate the target function and add it to the pool of
            # evaluated atoms structures.
            eval_atoms = Atoms(test_atoms, positions=positions_to_evaluate,
                               calculator=self.ase_calc)
            eval_atoms.get_potential_energy(force_consistent=self.force_consistent)
            self.images_pool += [copy.deepcopy(eval_atoms)]
            self.function_calls += 1
            self.force_calls += 1

            # 7. Print output.
            parprint('LGPMin:', self.function_calls,
                     time.strftime("%H:%M:%S", time.localtime()),
                     self.images_pool[-1].get_potential_energy(
                                    force_consistent=self.force_consistent),
                     get_fmax(self.images_pool[-1]))

        # Print final output when the surrogate is converged.
        print_cite_min()
        parprint('The optimization can be found in:',
                 trajectory.split('.')[0] + '_experiences.traj')
