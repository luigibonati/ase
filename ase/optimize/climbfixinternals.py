import numpy as np
from numpy.linalg import eigh

from ase.optimize.bfgs import BFGS
from ase.constraints import FixInternals


class ClimbFixInternals(BFGS):
    """
    Class for transition state search and optimization
    --------------------------------------------------

    Climbs the 1D reaction coordinate defined as constrained internal coordinate
    via the :class:`~ase.constraints.FixInternals` class while minimizing all
    remaining degrees of freedom.

    Two optimizers, 'A' and 'B', are applied orthogonal to each other.
    Optimizer 'A' climbs the constrained coordinate while optimizer 'B'
    optimizes the remaining degrees of freedom after each climbing step.

    Optimizer 'A' uses the BFGS algorithm to climb along the projected force of
    the selected constraint. Optimizer 'B' can be user-defined (default: BFGS).

    In combination with other constraints, the order of constraints matters.
    Generally, the FixInternals constraint should come last in the list of
    constraints, i.e., `atoms.set_constraint(list_of_constraints)`.
    This has been tested with the :class:`~ase.constraints.FixAtoms` constraint.

    Inspired by concepts described by Plessow [1]_, implemented by J. Amsler.

    .. [1] P. N. Plessow, Efficient Transition State Optimization of Periodic
           Structures through Automated Relaxed Potential Energy Surface Scans.
           J. Chem. Theory Comput. 2018, 14 (2), 981–990.
           https://doi.org/10.1021/acs.jctc.7b01070.
    """
    def __init__(self, atoms, restart=None, logfile='-', trajectory=None,
                 maxstep=None, master=None, alpha=None,
                 climb_coordinate=None,
                 optB=BFGS, optB_kwargs=None, optB_fmax=0.05):
        # auto_thresh=True, fixed_conv_ratio=0.8, max_interval_steps=3,
        # interval_step=0.5, adaptive_thresh=0.6, linear_interpol=False,
        # cubic=None):
        """
        Initialize like the parent class :class:`~ase.optimize.bfgs.BFGS`
        with the following additional parameters.

        Parameters
        ----------
        climb_coordinate: list
            Specifies which subconstraint of the
            :class:`~ase.constraints.FixInternals` constraint is to be climbed.
            Provide the 'constraint name' and corresponding indices as a list
            (without coefficients in the case of combo constraints).
            Examples:
            * `['FixBondLengthAlt', [[0, 1]]]`
            * `['FixAngle', [[0, 1, 2]]]`
            * `['FixDihedral', [[0, 1, 2, 3]]]`
            * `['FixBondCombo', [[0, 1], [2, 3]]]`
            * `['FixAngleCombo', [[0, 1, 2], [3, 4, 5]]]`
            * `['FixDihedralCombo', [[0, 1, 2, 3], [4, 5, 6, 7]]]`
    
        optB: any ASE optimizer, optional
            Optimizer 'B' for optimization of the remaining degrees of freedom.
            Default: :class:`~ase.optimize.bfgs.BFGS`
    
        optB_kwargs: dict, optional
            Specifies keyword arguments to be passed to optimizer 'B' at its
            initialization.
            Default: {'logfile': 'optB_{...}.log'} where {...} is the current
            value of the coordinate to be climbed
    
        optB_fmax: float, optional
            Specifies the convergence criterion `fmax` of optimizer 'B'.
        """
        BFGS.__init__(self, atoms, restart, logfile, trajectory,
                      maxstep, master, alpha)

        self.constr2climb = self.get_constr2climb(self.atoms, climb_coordinate)
        self.targetvalue = self.constr2climb.targetvalue

        self.optB = optB
        self.optB_kwargs = optB_kwargs or {}
        self.optB_fmax = optB_fmax
        self.optB_autolog = False if 'logfile' in self.optB_kwargs else True

    def get_constr2climb(self, atoms, climb_coordinate):
        atoms.set_positions(atoms.get_positions())  # initialize FixInternals
        available_constraint_types = list(map(type, atoms.constraints))
        index = available_constraint_types.index(FixInternals)
        for subconstr in atoms.constraints[index].constraints:
            if repr(subconstr).startswith(climb_coordinate[0]):
                if subconstr.indices == climb_coordinate[1]:
                    return subconstr
        raise ValueError('Given `climb_coordinate` not found on Atoms object.')

    def initialize(self):
        BFGS.initialize(self)
        self.projected_forces = None

    def read(self):
        (self.H, self.r0, self.f0, self.maxstep,
         self.projected_forces, self.targetvalue) = self.load()
        self.constr2climb.targetvalue = self.targetvalue  # update constr.

    def step(self, f=None):
        atoms = self.atoms
        f = self.get_projected_forces()  # initially computed during self.log()

        # similar to BFGS.step()
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

        if self.optB_autolog:
            logfilename = 'optB_{}.log'.format(self.targetvalue)
            self.optB_kwargs['logfile'] = logfilename
        optB = self.optB(atoms, **self.optB_kwargs)  # optimize remaining...
        optB.run(self.optB_fmax)                     # ...degrees of freedom

        self.projected_forces = self.get_projected_forces()

        self.dump((self.H, self.r0, self.f0, self.maxstep,
                   self.projected_forces, self.targetvalue))

    def get_projected_forces(self):
        f = self.constr2climb.projected_force * self.constr2climb.jacobian
        f = -1 * f.reshape(self.atoms.get_positions().shape)
        return f

    def converged(self):  # converge projected_forces
        forces = self.projected_forces
        return BFGS.converged(self, forces=forces)

    def log(self):
        forces = self.projected_forces
        if forces is None:  # always log fmax(projected_forces)
            self.atoms.get_forces()  # compute projected_forces
            forces = self.get_projected_forces()
        BFGS.log(self, forces=forces)
