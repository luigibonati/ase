import numpy as np
import time
import copy
import os
from ase.calculators.gp.calculator import GPCalculator
from ase import io, units
from ase.parallel import parprint, parallel_function
from ase.optimize.activelearning.acquisition import acquisition
from ase.md import VelocityVerlet
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.optimize import *
from scipy.signal import find_peaks
from scipy.spatial.distance import euclidean
from ase.visualize import view



class GPHopping:

    def __init__(self, atoms, calculator, model_calculator=None,
                 force_consistent=None,
                 trajectory='GPHopping.traj',
                 T0=1000., beta1=1.1, beta2=0.98,
                 energy_threshold=2., geometry_threshold=0.5, mdmin=1,
                 timestep=1.0, maxtime=1000., maxstep=0.4, maxoptsteps=1000,
                 optimizer=QuasiNewton):
        """
        Parameters
        --------------
        atoms: Atoms object
            The Atoms object to relax.

        model_calculator: Model object.
            Model calculator to be used for predicting the potential energy
            surface. The default is None which uses a GP model with the Squared
            Exponential Kernel and other default parameters. See
            *ase.calculator.gp.calculator* GPModel for default GP parameters.

        force_consistent: boolean or None
            Use force-consistent energy calls (as opposed to the energy
            extrapolated to 0 K). By default (force_consistent=None) uses
            force-consistent energies if available in the calculator, but
            falls back on force_consistent=False if not.

                trajectory: string
            Filename to store the predicted optimization.
                Additional information:
                - Energy uncertain: The energy uncertainty in each image can be
                  accessed in image.info['uncertainty'].

        """

        # MD simulation parameters.
        self.T0 = T0
        self.energy_threshold = energy_threshold
        self.mdmin = mdmin
        self.timestep = timestep
        self.maxtime = maxtime
        self.maxstep = maxstep
        self.maxoptsteps = maxoptsteps

        # Hopping parameters.
        self.beta1 = beta1
        self.beta2 = beta2
        self.geometry_threshold = geometry_threshold
        self.optimizer = optimizer

        # Model parameters.
        self.model_calculator = model_calculator
        # Default GP Calculator parameters if not specified by the user.
        if model_calculator is None:
            self.model_calculator = GPCalculator(train_images=[])
        self.model_calculator.calculate_uncertainty = True

        # Active Learning setup (single-point calculations).
        self.function_calls = 0
        self.force_calls = 0
        self.ase_calc = calculator
        self.atoms = atoms

        self.constraints = self.atoms.constraints
        self.force_consistent = force_consistent
        self.trajectory = trajectory

    def run(self, fmax=0.05, steps=1000):

        """
        Executing run will start the optimization process.

        Parameters
        ----------
        fmax : float
            Convergence criteria (in eV/Angstrom).

        steps: int
            Maximum number of steps for the surrogate.

        max_step: int
            Maximum uncertainty that is accepted before stopping the MD
            simulation.


        Returns
        -------
        Optimized structure. The optimization process can be followed in
        *trajectory_observations.traj*.

        """
        trajectory_main = self.trajectory.split('.')[0]
        trajectory_observations = trajectory_main + '_observations.traj'
        trajectory_candidates = trajectory_main + '_candidates.traj'
        trajectory_minima = trajectory_main + '_found_minima.traj'

        self.atoms.get_potential_energy(force_consistent=self.force_consistent)
        self.atoms.get_forces()
        dump_trajectory(atoms=self.atoms,
                        filename=trajectory_observations,
                        restart=True)
        train_images = io.read(trajectory_observations, ':')

        # Dump to trajectory_minima file the observations that satisfy fmax.
        if os.path.exists(trajectory_minima):
            os.remove(trajectory_minima)
        for i in train_images:
            if get_fmax(i) <= fmax:
                dump_trajectory(atoms=i,
                                filename=trajectory_minima,
                                restart=True)
        #############################
        # Make a step far from the minima in all directions (inverse gradient).
        if len(train_images) < 2:
            grad_step_size = 100.
            forces_initialize = self.atoms.get_forces().reshape(-1)
            positions_initialize = self.atoms.get_positions().reshape(-1)
            positions_initialize += grad_step_size * forces_initialize
            self.atoms.positions = positions_initialize.reshape(-1, 3)
            self.atoms.get_potential_energy(force_consistent=self.force_consistent)
            self.atoms.get_forces()
            dump_trajectory(atoms=self.atoms,
                            filename=trajectory_observations,
                            restart=True)
            train_images = io.read(trajectory_observations, ':')
        ##############################

        ml_steps = 0
        while ml_steps < steps:
            # 1. Collect observations.
            train_images = io.read(trajectory_observations, ':')
            # 2. Update GP calculator.
            gp_calc = copy.deepcopy(self.model_calculator)
            gp_calc.update_train_data(train_images=train_images)

            ################
            # prev_atoms = io.read(trajectory_observations, ':')
            # e_prev = []
            # for i in prev_atoms:
            #     e_prev.append(i.get_potential_energy())
            #
            # parprint('Starting from prev geometry number: ', np.argmin(e_prev))
            # self.atoms = prev_atoms[np.argmin(e_prev)]

            parprint('Starting from last minima found...')
            prev_minima = io.read(trajectory_minima, ':')
            self.atoms = io.read(trajectory_minima, 0)

            # 3. Perform MD simulation in the predicted PES.
            self.atoms.set_calculator(gp_calc)
            self.atoms.get_potential_energy()

            # 3.1 Optimize output from MD simulation to find candidates.

            temperature = self.T0
            candidates = []
            np.random.seed(len(prev_minima))
            while len(candidates) < 1:
                md_results = mdsim(atoms=self.atoms,
                                   temperature=temperature,
                                   maxstep=self.maxstep,
                                   timestep=self.timestep,
                                   maxtime=self.maxtime,
                                   mdmin=self.mdmin,
                                   energy_threshold=self.energy_threshold,
                                   trajectory='md_simulation.traj')

                parprint('Stopping reason:', md_results['stop_reason'])

                # Optimize all MD minima found.
                for index in md_results['indexes_minima']:
                    self.atoms.positions = md_results['positions'][index]
                    optimize_atoms(atoms=self.atoms,
                                   optimizer=self.optimizer,
                                   fmax=fmax * 0.01,
                                   maxstep=self.maxstep,
                                   maxoptsteps=self.maxoptsteps)
                    candidates += [copy.deepcopy(self.atoms)]

                # Remove candidates that are close to other previously found
                # minima (within a geometry_threshold).
                filtered_candidates = []
                for i in range(0, len(candidates)):
                    unique_candidate = True
                    pos_i = candidates[i].get_positions().reshape(-1)
                    for j in range(0, len(prev_minima)):
                        pos_j = prev_minima[j].get_positions().reshape(-1)
                        dmax = np.max(euclidean(pos_i, pos_j))
                        if dmax < self.geometry_threshold:
                            unique_candidate = False
                            temperature *= self.beta1  # Increase T.
                            if j == len(prev_minima):
                                parprint('Re-found last minimum.')
                            else:
                                parprint('Re-found minima.')
                        if unique_candidate is True:
                            filtered_candidates += [candidates[i]]
                            temperature *= self.beta2  # Decrease T.
                candidates = copy.deepcopy(filtered_candidates)

                if md_results['stop_reason'] == 'max_uncertainty_reached':
                    candidates += [copy.deepcopy(self.atoms)]
                if md_results['stop_reason'] == 'max_time_reached':
                    parprint('Increasing temp. due to max. time reached.')
                    temperature *= self.beta1  # Increase temperature.
                if md_results['stop_reason'] == 'max_energy_reached':
                    parprint('Decreasing temp. due to max. energy reached.')
                    temperature *= self.beta2  # Decrease temperature.
                parprint('Current initial temperature:', temperature)

            # 4. Order candidates using acquisition function:
            sorted_candidates = acquisition(train_images=train_images,
                                            candidates=candidates,
                                            mode='energy',
                                            objective='min')

            # Select the best candidate.
            best_candidate = sorted_candidates.pop(0)

            view(best_candidate)

            # Save the other candidates for multi-task optimization.
            io.write(trajectory_candidates, sorted_candidates)

            # 5. Evaluate the target function and save it in *observations*.
            self.atoms.positions = best_candidate.get_positions()
            self.atoms.set_calculator(self.ase_calc)
            self.atoms.get_potential_energy(force_consistent=self.force_consistent)
            self.atoms.get_forces()
            dump_trajectory(atoms=self.atoms,
                            filename=trajectory_observations,
                            restart=True)
            self.function_calls += 1
            self.force_calls += 1

            # If the last evaluated structure is below fmax:
            if get_fmax(self.atoms) <= fmax:
                dump_trajectory(atoms=self.atoms,
                                filename=trajectory_minima, restart=True)
            # 6. Print output.
            msg = "--------------------------------------------------------"
            parprint(msg)
            parprint('Step:', self.function_calls)
            parprint('MH Steps:', ml_steps)
            parprint('Time:', time.strftime("%m/%d/%Y, %H:%M:%S",
                                            time.localtime()))
            parprint('Energy', self.atoms.get_potential_energy(self.force_consistent))
            parprint("fmax:", get_fmax(train_images[-1]))
            parprint("Minima found:", len(io.read(trajectory_minima, ':')))
            msg = "--------------------------------------------------------\n"
            parprint(msg)

@parallel_function
def optimize_atoms(atoms, optimizer, fmax, maxstep=0.4, maxoptsteps=1000,
                   trajectory=None, logfile=None):
    atoms.get_potential_energy()
    opt = optimizer(atoms=atoms, logfile=logfile, trajectory=trajectory)
    converged = False
    step = 0
    while converged is False:
        opt.run(fmax=fmax, steps=1)
        if get_fmax(atoms) <= fmax or step > maxoptsteps or \
                atoms.get_calculator().results['uncertainty'] >= maxstep:
            converged = True
        else:
            step += 1
    return atoms


@parallel_function
def mdsim(atoms, temperature, maxstep=0.4, timestep=1.0, maxtime=2000.,
          mdmin=1, energy_threshold=2.0, trajectory='md_simulation.traj'):
    """Performs a molecular dynamics simulation until any of the following
    conditions is achieved: (A) Max uncertainty reached (determined by
    'max_step') or (B) 'mdmin' number of minima are found or (C)
    'max_time' (in fs) has been reached or (D) energy threshold has been
    crossed over ('energy_threshold')."""

    initial_energy = atoms.get_potential_energy()  # Initial energy.
    current_time = 0.0  # Initial time.
    energies, uncertainties, positions, indexes_minima = [], [], [], []
    stop_reason = None

    MaxwellBoltzmannDistribution(atoms,
                                 temp=temperature * units.kB,
                                 force_temp=True)
    dyn = VelocityVerlet(atoms, timestep=timestep * units.fs,
                         trajectory=trajectory)

    while stop_reason is None:
        energies.append(atoms.get_potential_energy())
        positions.append(atoms.positions)
        uncertainties.append(atoms.get_calculator().results['uncertainty'])
        indexes_minima = find_peaks(-np.array(energies))[0]

        # A. Stop MD simulation if max. uncertainty is crossed over.
        if uncertainties[-1] > maxstep:
            stop_reason = 'max_uncertainty_reached'
        # B. Stop MD simulation if 'mdmin' number of minima are found.
        elif len(indexes_minima) >= mdmin:
            stop_reason = 'mdmin_found'
        # C. Stop if max. time has been reached.
        elif current_time >= maxtime:
            stop_reason = 'max_time_reached'
        # D) Stop if energy threshold has been crossed over (e_threshold).
        elif energies[-1] >= initial_energy + energy_threshold:
            stop_reason = 'max_energy_reached'
        else:
            dyn.run(1)  # Run MD step.
            current_time += timestep  # Update the time of the MD simulation.
    indexes_maxima = find_peaks(np.array(energies))[0]
    results = {'stop_reason': stop_reason, 'indexes_minima': indexes_minima,
               'indexes_maxima': indexes_maxima, 'energies': energies,
               'uncertainties': uncertainties, 'positions': positions}
    return results


@parallel_function
def get_fmax(atoms):
    """
    Returns fmax for a given atoms structure.
    """
    forces = atoms.get_forces()
    return np.sqrt((forces**2).sum(axis=1).max())


@parallel_function
def dump_trajectory(atoms, filename, restart):
    """
    Saves a trajectory file containing the atoms observations.

    Parameters
    ----------
    atoms: object
        Atoms object to be appended to previous observations.
    filename: string
        Name of the trajectory file to save the observations.
    restart: boolean
        Append mode (true or false).
     """

    if restart is True:
        try:
            prev_atoms = io.read(filename, ':')  # Actively searching.
            if atoms not in prev_atoms:  # Avoid duplicates.
                # Update observations.
                new_atoms = prev_atoms + [atoms]
                io.write(filename=filename, images=new_atoms)
        except Exception:
            io.write(filename=filename, images=atoms, append=False)
    if restart is False:
        io.write(filename=filename, images=atoms, append=False)


class ComparePositions:
    """Class that compares the atomic positions between two ASE atoms
    objects. Returns the maximum distance that any atom has moved, assuming
    all atoms of the same element are indistinguishable. If translate is
    set to True, allows for arbitrary translations within the unit cell,
    as well as translations across any periodic boundary conditions. When
    called, returns the maximum displacement of any one atom."""

    def __init__(self, translate=True):
        self._translate = translate

    def __call__(self, atoms1, atoms2):
        atoms1 = atoms1.copy()
        atoms2 = atoms2.copy()
        if not self._translate:
            dmax = self. _indistinguishable_compare(atoms1, atoms2)
        else:
            dmax = self._translated_compare(atoms1, atoms2)
        return dmax

    def _translated_compare(self, atoms1, atoms2):
        """Moves the atoms around and tries to pair up atoms, assuming any
        atoms with the same symbol are indistinguishable, and honors
        periodic boundary conditions (for example, so that an atom at
        (0.1, 0., 0.) correctly is found to be close to an atom at
        (7.9, 0., 0.) if the atoms are in an orthorhombic cell with
        x-dimension of 8. Returns dmax, the maximum distance between any
        two atoms in the optimal configuration."""
        atoms1.set_constraint()
        atoms2.set_constraint()
        for index in range(3):
            assert atoms1.pbc[index] == atoms2.pbc[index]
        least = self._get_least_common(atoms1)
        indices1 = [atom.index for atom in atoms1 if atom.symbol == least[0]]
        indices2 = [atom.index for atom in atoms2 if atom.symbol == least[0]]
        # Make comparison sets from atoms2, which contain repeated atoms in
        # all pbc's and bring the atom listed in indices2 to (0,0,0)
        comparisons = []
        repeat = []
        for bc in atoms2.pbc:
            if bc:
                repeat.append(3)
            else:
                repeat.append(1)
        repeated = atoms2.repeat(repeat)
        moved_cell = atoms2.cell * atoms2.pbc
        for moved in moved_cell:
            repeated.translate(-moved)
        repeated.set_cell(atoms2.cell)
        for index in indices2:
            comparison = repeated.copy()
            comparison.translate(-atoms2[index].position)
            comparisons.append(comparison)
        # Bring the atom listed in indices1 to (0,0,0) [not whole list]
        standard = atoms1.copy()
        standard.translate(-atoms1[indices1[0]].position)
        # Compare the standard to the comparison sets.
        dmaxes = []
        for comparison in comparisons:
            dmax = self._indistinguishable_compare(standard, comparison)
            dmaxes.append(dmax)
        return min(dmaxes)

    def _get_least_common(self, atoms):
        """Returns the least common element in atoms. If more than one,
        returns the first encountered."""
        symbols = [atom.symbol for atom in atoms]
        least = ['', np.inf]
        for element in set(symbols):
            count = symbols.count(element)
            if count < least[1]:
                least = [element, count]
        return least

    def _indistinguishable_compare(self, atoms1, atoms2):
        """Finds each atom in atoms1's nearest neighbor with the same
        chemical symbol in atoms2. Return dmax, the farthest distance an
        individual atom differs by."""
        atoms2 = atoms2.copy()  # allow deletion
        atoms2.set_constraint()
        dmax = 0.
        for atom1 in atoms1:
            closest = [np.nan, np.inf]
            for index, atom2 in enumerate(atoms2):
                if atom2.symbol == atom1.symbol:
                    d = np.linalg.norm(atom1.position - atom2.position)
                    if d < closest[1]:
                        closest = [index, d]
            if closest[1] > dmax:
                dmax = closest[1]
            del atoms2[closest[0]]
        return dmax
