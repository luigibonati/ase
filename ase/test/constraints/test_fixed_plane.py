from ase.build import molecule
from ase.calculators.emt import EMT
from ase.constraints import FixedPlane
from ase.optimize import BFGS
import numpy as np
import pytest


@pytest.mark.parametrize('indices', [0, [0], [0, 1]])
def test_valid_inputs(indices):
    c = FixedPlane(indices, [1, 0, 0])
