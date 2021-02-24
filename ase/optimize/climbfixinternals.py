import numpy as np
from numpy.linalg import eigh

from ase.optimize.bfgs import BFGS
from ase.constraints import FixInternals


class ClimbFixInternals(BFGS):
    """
    Class for transition state search and optimization

    Climbs the 1D reaction coordinate defined as constrained internal coordinate
    via the :class:`~ase.constraints.FixInternals` class while minimizing all
    remaining degrees of freedom.

    Two optimizers, 'A' and 'B', are applied orthogonal to each other.
    Optimizer 'A' climbs the constrained coordinate while optimizer 'B'
    optimizes the remaining degrees of freedom after each climbing step.

    Optimizer 'A' uses the BFGS algorithm to climb along the projected force of
    the selected constraint. Optimizer 'B' can be user-defined (default: BFGS).

    In combination with other constraints, the order of constraints matters.
    Generally, the FixInternals constraint should come first in the list of
    constraints, i.e., `atoms.set_constraint(list_of_constraints)`.
    This has been tested with the :class:`~ase.constraints.FixAtoms` constraint.

    Inspired by concepts described by P. N. Plessow [1]_,
    implemented by J. Amsler.

    .. [1] P. N. Plessow, Efficient Transition State Optimization of Periodic
           Structures through Automated Relaxed Potential Energy Surface Scans.
           J. Chem. Theory Comput. 2018, 14 (2), 981–990.
           https://doi.org/10.1021/acs.jctc.7b01070.

    .. note::
       Convergence is based on 'fmax' of the total forces, i.e. on 'fmax' of
       the sum of the projected forces and the forces of the remaining degrees
       of freedom.

    Example
    -------
    >>> from ase.constraints import FixInternals
    >>> from ase.optimize.climbfixinternals import ClimbFixInternals
    >>> # define the reaction coordinate as a linear combination of bond lengths
    >>> # 1.0 * bond(0,1) -1.0 * bond(2,3)
    >>> reaction_coordinate = [[0, 1, 1.0], [2, 3, -1.0]]
    >>> constr = FixInternals(bondcombos=[[None, bond_combo]])
    >>> # 'None' automatically takes initial constraint value as current value
    >>> atoms.set_constraint(constr)  # 'atoms' is an ASE Atoms object
    >>> dyn = ClimbFixInternals(atoms, climb_coordinate=reaction_coordinate)
    >>> dyn.run()  # climbs the reaction coord. while relaxing everything else
    """
    def __init__(self, atoms, restart=None, logfile='-', trajectory=None,
                 maxstep=None, master=None, alpha=None,
                 climb_coordinate=None,
                 optB=BFGS, optB_kwargs=None, optB_fmax=0.05,
                 optB_fmax_scaling=0.0):
        """
        Initialize like the parent class :class:`~ase.optimize.bfgs.BFGS`
        with the following additional parameters.

        Parameters
        ----------
        climb_coordinate: list
            Specifies which subconstraint of the
            :class:`~ase.constraints.FixInternals` constraint is to be climbed.
            Provide the corresponding nested list of indices
            (including coefficients in the case of Combo constraints).
            Examples:
            * `[0, 1]` defines a constrained bond
            * `[[0, 1, 1.0], [2, 3, -1.0]]` defines a constrained linear
            combination of bond lengths

        optB: any ASE optimizer, optional
            Optimizer 'B' for optimization of the remaining degrees of freedom.
            Default: :class:`~ase.optimize.bfgs.BFGS`

        optB_kwargs: dict, optional
            Specifies keyword arguments to be passed to optimizer 'B' at its
            initialization. By default, optimizer 'B' writes a logfile and
            trajectory (optB_{...}.log, optB_{...}.traj) where {...} is the
            current value of the `climb_coordinate`. Set `logfile` to '-' for
            console output. Set `trajectory` to 'None' to suppress writing of
            the trajectory file.
            Default: None

        optB_fmax: float, optional
            Specifies the convergence criterion `fmax` of optimizer 'B'.

        optB_fmax_scaling: float, optional
            Scaling factor to dynamically tighten `fmax` of optimizer 'B' to
            the value of `optB_fmax` when close to convergence.
            Can speed up the climbing process. The scaling formula is

            `fmax = optB_fmax + optB_fmax_scaling * projected_force`

            The final optimization with optimizer 'B' is
            performed with `optB_fmax` independent of `optB_fmax_scaling`.
            Recommended values are below 1.0.
            Default: 0.0
        """
        self.targetvalue = None  # may be assigned during restart in self.read()
        BFGS.__init__(self, atoms, restart, logfile, trajectory,
                      maxstep, master, alpha)

        self.constr2climb = self.get_constr2climb(self.atoms, climb_coordinate)
        if self.targetvalue is None:  # if not assigned during restart
            self.targetvalue = self.constr2climb.targetvalue

        self.optB = optB
        self.optB_kwargs = optB_kwargs or {}
        self.optB_fmax = optB_fmax
        self.scaling = optB_fmax_scaling

    def get_constr2climb(self, atoms, climb_coordinate):
        """Get pointer to the subconstraint that is to be climbed.
        Identification by its definition via indices (and coefficients)."""
        atoms.set_positions(atoms.get_positions())  # initialize FixInternals
        available_constraint_types = list(map(type, atoms.constraints))
        index = available_constraint_types.index(FixInternals)  # locate constr.
        for subconstr in atoms.constraints[index].constraints:
            if 'Combo' in repr(subconstr):
                defin = [d + [c] for d, c in zip(subconstr.indices,
                                                 subconstr.coefs)]
                if defin == climb_coordinate:  # identify Combo constraints...
                    return subconstr  # ...by combination of indices and coefs.
            else:  # identify primitive constraints by their indices
                if subconstr.indices == [climb_coordinate]:
                    return subconstr
        raise ValueError('Given `climb_coordinate` not found on Atoms object.')

    def read(self):
        self.H, self.r0, self.f0, self.maxstep, self.targetvalue = self.load()

    def step(self, f=None):
        atoms = self.atoms

        # setup optimizer 'B'
        if 'logfile' not in self.optB_kwargs:  # autologging
            logfilename = 'optB_{}.log'.format(self.targetvalue)
            self.optB_kwargs['logfile'] = logfilename
        if 'trajectory' not in self.optB_kwargs:  # autologging
            trajfilename = 'optB_{}.traj'.format(self.targetvalue)
            self.optB_kwargs['trajectory'] = trajfilename
        optB = self.optB(atoms, **self.optB_kwargs)

        # initial relaxation of remaining degrees of freedom with optimizer 'B'
        if self.nsteps == 0:
            optB.run(self.get_scaled_fmax())  # optimize with scaled fmax

        f = self.get_projected_forces()  # get directions for climbing
        # climb with optimizer 'A', similar to BFGS.step()
        r = atoms.get_positions()
        f = f.reshape(-1)
        self.update(r.flat, f, self.r0, self.f0)
        omega, V = eigh(self.H)
        dr = np.dot(V, np.dot(f, V) / np.fabs(omega)).reshape((-1, 3))
        steplengths = (dr**2).sum(1)**0.5
        dr = self.determine_step(dr, steplengths)

        self.constr2climb.adjust_positions(r, r + dr)  # update constr.sigma
        self.targetvalue += self.constr2climb.sigma  # climb the constraint
        self.constr2climb.targetvalue = self.targetvalue  # adjust positions...
        atoms.set_positions(atoms.get_positions())        # ...to targetvalue

        self.r0 = r.flat.copy()
        self.f0 = f.copy()

        # optimize remaining degrees of freedom with optimizer 'B'
        fmax = self.get_scaled_fmax()
        optB.run(fmax)  # optimize with scaled fmax
        if self.converged() and fmax > self.optB_fmax:
            optB.run(self.optB_fmax)  # final optimization with desired fmax

        self.dump((self.H, self.r0, self.f0, self.maxstep, self.targetvalue))

    def get_scaled_fmax(self):
        """Determine adaptive 'fmax' based on the estimated distance to the
        transition state."""
        return self.optB_fmax + self.scaling * self.constr2climb.projected_force

    def get_projected_forces(self):  # get projected forces in uphill direction
        f = self.constr2climb.projected_force * self.constr2climb.jacobian
        f = -1 * f.reshape(self.atoms.positions.shape)
        return f

    def get_total_forces(self, forces=None):
        forces = forces or self.atoms.get_forces()
        forces += self.get_projected_forces()
        return forces

    def converged(self, forces=None):
        """Did the optimization converge based on the total forces?"""
        forces = self.get_total_forces(forces)
        return BFGS.converged(self, forces=forces)

    def log(self, forces=None):
        forces = self.get_total_forces(forces)
        BFGS.log(self, forces=forces)
