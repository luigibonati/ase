import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.lj import LennardJones
from ase.constraints import UnitCellFilter
from ase.optimize import BFGS

# Theoretical infinite-cutoff LJ FCC unit cell parameters
vol0 = 4 * 0.91615977036  # theoretical minimum
a0 = vol0**(1 / 3)


@pytest.fixture
def atoms():
    """two atoms at potential minimum"""
    a = bulk('X', 'fcc', a=a0)

    a.calc = LennardJones()

    return a


def test_stress_voigt_shape(atoms):
    a = atoms
    # test voigt shape
    for ideal_gas in (False, True):
        kw = {'include_ideal_gas': ideal_gas}

        assert a.get_stress(voigt=True, **kw).shape == (6,)
        assert a.get_stress(voigt=False, **kw).shape == (3, 3)

        assert a.get_stresses(voigt=True, **kw).shape == (len(a), 6)
        assert a.get_stresses(voigt=False, **kw).shape == (len(a), 3, 3)


@pytest.mark.slow
def test_stress(atoms):
    a = atoms
    cell0 = a.get_cell()

    a.set_cell(np.dot(a.cell,
                      [[1.02, 0, 0.03],
                       [0, 0.99, -0.02],
                       [0.1, -0.01, 1.03]]),
               scale_atoms=True)

    a *= (1, 2, 3)
    cell0 *= np.array([1, 2, 3])[:, np.newaxis]

    a.rattle()

    # Verify analytical stress tensor against numerical value
    s_analytical = a.get_stress()
    s_numerical = a.calc.calculate_numerical_stress(a, 1e-5)
    s_p_err = 100 * (s_numerical - s_analytical) / s_numerical

    print("Analytical stress:\n", s_analytical)
    print("Numerical stress:\n", s_numerical)
    print("Percent error in stress:\n", s_p_err)
    assert np.all(abs(s_p_err) < 1e-5)

    # Minimize unit cell
    opt = BFGS(UnitCellFilter(a))
    opt.run(fmax=1e-3)

    # Verify minimized unit cell using Niggli tensors
    g_minimized = np.dot(a.cell, a.cell.T)
    g_theory = np.dot(cell0, cell0.T)
    g_p_err = 100 * (g_minimized - g_theory) / g_theory

    print("Minimized Niggli tensor:\n", g_minimized)
    print("Theoretical Niggli tensor:\n", g_theory)
    print("Percent error in Niggli tensor:\n", g_p_err)
    assert np.all(abs(g_p_err) < 1)
