from ase.constraints import FixedLine
from ase.build import molecule
from ase.constraints import FixedLine
from ase.calculators.emt import EMT
from ase.optimize import BFGS
import pytest
import numpy as np


@pytest.mark.parametrize('indices', [0, [0], [0, 1]])
def test_valid_inputs(indices):
    c = FixedLine(indices, [1, 0, 0])
