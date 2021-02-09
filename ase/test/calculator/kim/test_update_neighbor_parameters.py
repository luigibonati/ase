import numpy as np
import pytest
from pytest import mark
from ase import Atoms


@mark.calculator_lite
def test_update_neighbor_parameters(KIM):
    """
    Check that the neighbor parameters are updated properly when model parameter
    is updated. This is done by instantiating the calculator for a specific
    Lennard-Jones (LJ) potential, included with the KIM API, for Mo-Mo
    interactions. An Mo dimer is constructed with a random, but small,
    separation. The energy is then computed as a reference. Then an Mo trimer is
    constructed, by adding an atom near, but still within, the cutoff. The
    energy of the trimer is computed and asserted to be different than the
    energy of the dimer. Then, the cutoff parameter is modified to exclude the
    atom on the far side and the energy of the trimer is computed again. The
    final energy is asserted to be approximately equal to the energy of the
    dimer.
    """

    # Create KIM calculator
    calc = KIM("LennardJones612_UniversalShifted__MO_959249795837_003")
    index = 4879  # Index for Mo-Mo interaction
    cutoff = calc.get_parameters(cutoffs=index)["cutoffs"][1]
    calc.set_parameters(shift=[0, 0])  # Disable energy shifting

    # Define separation distance
    closer_separation = np.random.RandomState(11).uniform(0.25 * cutoff, 0.35 * cutoff)
    farther_separation = np.random.RandomState(11).uniform(0.8 * cutoff, 0.9 * cutoff)
    separations = [closer_separation, farther_separation]
    positions_trimer = [
        [0, 0, 0],
        [0, 0, separations[0]],
        [0, 0, separations[1]],
    ]

    # Create a dimer
    dimer = Atoms("Mo" * 2, positions=positions_trimer[:2])
    dimer.calc = calc

    # Create a trimer
    trimer = Atoms("Mo" * 3, positions=positions_trimer)
    trimer.calc = calc

    # Energy of the dimer
    eng_dimer = dimer.get_potential_energy()

    # Original energy of the trimer
    eng_trimer_orig = trimer.get_potential_energy()

    # Update the cutoff parameter
    scaling_factor = 0.4
    calc.set_parameters(cutoffs=[index, cutoff * scaling_factor])

    # Energy of the trimer after modifying cutoff
    eng_trimer_modified = trimer.get_potential_energy()

    # Check if the far atom is included when the original cutoff is used
    assert eng_trimer_orig != pytest.approx(eng_dimer, rel=1e-4)

    # Check if the far atom is excluded when the modified cutoff is used
    assert eng_trimer_modified == pytest.approx(eng_dimer, rel=1e-4)
