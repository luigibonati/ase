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

@pytest.mark.parametrize(
    'indices', [
        [0, 1, 1],
        [[0, 1], [0, 1]],
    ]
)
def test_invalid_inputs(indices):
    with pytest.raises(ValueError) as e_info:
        c = FixedLine(indices, [1, 0, 0])

@pytest.mark.parametrize('indices', [0, [0], [0, 1]])
def test_repr(indices):
    c = FixedLine(indices, [1, 0, 0])
    repr(FixedLine(indices, [1, 0, 0])) == (
        "<FixedLine: {'indices': " + str(indices) + ", 'direction': [1. 0. 0.]}>"
    )
