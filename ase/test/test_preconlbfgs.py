import numpy as np
import pytest

from ase.build import bulk
from ase.calculators.emt import EMT
from ase.optimize.precon import Exp, PreconLBFGS, PreconFIRE, make_precon
from ase.constraints import FixBondLength, FixAtoms

@pytest.fixture
def setup_atoms():
    N = 1
    atoms = bulk('Cu', cubic=True)
    atoms *= (N, N, N)

    # perturb the atoms
    s = atoms.get_scaled_positions()
    s[:, 0] *= 0.995
    atoms.set_scaled_positions(s)
    atoms.set_calculator(EMT())

    # compute initial Hessian
    H0 = fd_hessian(atoms)
    return atoms, H0

def fd_hessian(atoms, dx=1e-5, precon=None):
    """
    Finite difference hessian from Jacobian of forces
    """
    hess = np.zeros((3*len(atoms), 3*len(atoms)))
    for i in range(len(atoms)):
        for j in range(3):
            atoms.positions[i, j] += dx
            fp = atoms.get_forces().reshape(-1)
            if precon is not None:
                fp = precon.solve(fp)
            atoms.positions[i, j] -= 2*dx
            fm = atoms.get_forces().reshape(-1)
            if precon is not None:
                fm = precon.solve(fm)
            atoms.positions[i, j] -= dx
            hess[3*i + j, :] = -(fp - fm)/(2*dx)
    # print(hess.round(4))
    return hess

@pytest.mark.parametrize('precon', ['Exp'])
@pytest.mark.filterwarnings('ignore:estimate_mu')
def test_make_precon(precon, setup_atoms):
    atoms, H0 = setup_atoms
    P = make_precon(precon)
    P.make_precon(atoms)
    HP = fd_hessian(atoms, precon=P)
    # check the preconditioner reduces condition number of Hessian
    assert np.linalg.cond(HP) < np.linalg.cond(H0)

@pytest.mark.slow
def test_preconlbfgs(setup_atoms):
    a0, H0 = setup_atoms
    nsteps = []
    energies = []
    for OPT in [PreconLBFGS, PreconFIRE]:
        for precon in [None, Exp(A=3, mu=1.0)]:
            atoms = a0.copy()
            atoms.set_calculator(EMT())
            opt = OPT(atoms, precon=precon, use_armijo=True)
            opt.run(1e-4)
            energies += [atoms.get_potential_energy()]
            nsteps += [opt.get_number_of_steps()]

    # check we get the expected energy for all methods
    assert np.abs(np.array(energies) - -0.022726045433998365).max() < 1e-4

    # test with fixed bondlength and fixed atom constraints
    cu0 = bulk("Cu") * (2, 2, 2)
    cu0.rattle(0.01)
    a0 = cu0.get_distance(0, 1)
    cons = [FixBondLength(0, 1), FixAtoms([2, 3])]
    for precon in [None, Exp(mu=1.0)]:
        cu = cu0.copy()
        cu.set_calculator(EMT())
        cu.set_distance(0, 1, a0*1.2)
        cu.set_constraint(cons)
        opt = PreconLBFGS(cu, precon=precon, use_armijo=True)
        opt.run(fmax=1e-3)

        assert abs(cu.get_distance(0, 1)/a0 - 1.2) < 1e-3
        assert np.all(abs(cu.positions[2] - cu0.positions[2]) < 1e-3)
        assert np.all(abs(cu.positions[3] - cu0.positions[3]) < 1e-3)
