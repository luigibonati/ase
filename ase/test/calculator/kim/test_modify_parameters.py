import numpy as np
import pytest
from pytest import mark
from ase import Atoms


@mark.calculator_lite
def test_modify_parameters(KIM):
    """
    Test that the parameters of the KIM calculator are correctly updated. This
    is done by first constructing a Mo dimer with random separation. Using the
    Lennard-Jones model, the energy is proportional to the value of epsilon. The
    energies of the system, computed using 2 different values of epsilon, are
    then compared. The ratio of these energies are equal to the ratio of the 2
    epsilon values that are used to compute them.
    """

    # In LennardJones612_UniversalShifted__MO_959249795837_003, the cutoff
    # for Mo interaction is 10.9759 Angstroms.
    cutoff = 10.9759

    # Create random dimer with separation < cutoff
    dimer_separation = np.random.uniform(0, cutoff)
    atoms = Atoms("Mo" * 2, positions=[[0, 0, 0], [0, 0, dimer_separation]])

    calc = KIM('LennardJones612_UniversalShifted__MO_959249795837_003')
    atoms.calc = calc

    # Retrieve the original energy scaling parameter
    eps_orig = calc.get_parameters(epsilons=4879)['epsilons'][1]  # eV

    # Get the energy using the original parameter as a reference value
    E_orig = atoms.get_potential_energy()  # eV

    # Scale the energy scaling parameter and set this value to the calculator
    energy_scaling_factor = 2.0
    eps_modified = energy_scaling_factor * eps_orig
    calc.set_parameters(epsilons=[4879, eps_modified])

    # Get the energy after modifying the parameter
    E_modified = atoms.get_potential_energy()  # eV

    assert E_modified == pytest.approx(energy_scaling_factor * E_orig, rel=1e-4)
