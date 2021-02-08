import numpy as np
import pytest
from pytest import mark
from ase import Atoms


@mark.calculator_lite
def test_update_neighbor_parameters(KIM):
    """
    Check that the neighbor parameters are updated properly when model parameter
    is updated. This is done by instantiating the calculator for a specific
    Lennard-Jones (LJ) potential, included with the KIM API, for Fr-Fr
    interactions. The skin and influence distance are proportional to the
    model's influence distance. While the cutoff are the sum of the model's
    cutoff with the skin.

    The values of the skin, and influence distance using the default parameter
    values are obtained. Then, the value of models cutoff is scaled by some
    scaling factor. Using the new model's, the values of skin, influence
    distance, and cutoff are compared to the original values. The ration between
    the new skin and influence distance to the original ones are asserted to be
    approximately equal to the scaling factor. The new cutoff is asserted to be
    approximately equal to the sum of the new model's cutoff and the new skin.
    """

    # Create KIM calculator
    calc = KIM("LennardJones612_UniversalShifted__MO_959249795837_003")
    index = 8299  # Index for Fr-Fr interaction
    model_cutoff_orig = calc.get_parameters(
        cutoffs=index
    )  # Original cutoff value

    # Get original neigh parameters
    skin_orig, influence_dist_orig, _ = _get_neigh_parameters(calc)

    # Update the cutoff parameter
    scaling_factor = 2
    model_cutoff_modified = scaling_factor * model_cutoff_orig["cutoffs"][1]
    calc.set_parameters(cutoffs=[index, model_cutoff_modified])
    (
        skin_modified,
        influence_dist_modified,
        cutoffs_modified,
    ) = _get_neigh_parameters(calc)

    # Check the skin
    assert skin_modified == pytest.approx(scaling_factor * skin_orig, rel=1e-4)

    # Check the influence distance
    assert influence_dist_modified == pytest.approx(
        scaling_factor * influence_dist_orig, rel=1e-4
    )

    # Check the cutoffs
    assert cutoffs_modified == pytest.approx(
        model_cutoff_modified + skin_modified, rel=1e-4
    )


def _get_neigh_parameters(calc):
    skin = calc.neigh.skin
    influence_dist = calc.neigh.influence_dist
    cutoffs = calc.neigh.cutoffs

    return skin, influence_dist, cutoffs
