from ase.build import molecule
from ase.calculators.emt import EMT
from ase.constraints import FixedLine
from ase.optimize import BFGS
import numpy as np
import pytest


@pytest.mark.parametrize(
    'indices', [
        0,
        [0],
        [0, 1],
        np.array([0, 1], dtype=np.int64),
    ]
)
def test_valid_inputs_indices(indices):
    _ = FixedLine(indices, [1, 0, 0])


@pytest.mark.parametrize(
    'indices', [
        [0, 1, 1],
        [[0, 1], [0, 1]],
    ]
)
def test_invalid_inputs_indices(indices):
    with pytest.raises(ValueError) as _:
        _ = FixedLine(indices, [1, 0, 0])


@pytest.mark.parametrize('direction', [[0, 0, 1], (0, 0, 1)])
def test_valid_inputs_direction(direction):
    _ = FixedLine(0, direction)


@pytest.mark.parametrize('direction', [[0, 1], None, "42"])
def test_invalid_inputs_direction(direction):
    with pytest.raises(Exception) as _:
        _ = FixedLine(0, direction)


@pytest.mark.parametrize('indices', [0, [0], [0, 1]])
def test_repr(indices):
    repr(FixedLine(indices, [1, 0, 0])) == (
        "<FixedLine: "
        "{'indices': " + str(indices) + ", 'direction': [1. 0. 0.]}>"
    )


def test_constrained_optimization_single():
    c = FixedLine(0, [1, 0, 0])

    mol = molecule("butadiene")
    mol.set_constraint(c)

    assert len(mol.constraints) == 1
    assert isinstance(c.dir, np.ndarray)
    assert (np.asarray([1, 0, 0]) == c.dir).all()

    mol.calc = EMT()

    cold_positions = mol[0].position.copy()
    opt = BFGS(mol)
    opt.run(steps=5)
    cnew_positions = mol[0].position.copy()

    assert np.max(np.abs(cnew_positions[1:] - cold_positions[1:])) < 1e-8
    assert np.max(np.abs(cnew_positions[0] - cold_positions[0])) > 1e-8


def test_constrained_optimization_multiple():
    indices = [0, 1]
    c = FixedLine(indices, [1, 0, 0])

    mol = molecule("butadiene")
    mol.set_constraint(c)

    assert len(mol.constraints) == 1
    assert isinstance(c.dir, np.ndarray)
    assert (np.asarray([1, 0, 0]) == c.dir).all()

    mol.calc = EMT()

    cold_positions = mol[indices].positions.copy()
    opt = BFGS(mol)
    opt.run(steps=5)
    cnew_positions = mol[indices].positions.copy()

    assert np.max(np.abs(cnew_positions[:, 1:] - cold_positions[:, 1:])) < 1e-8
    assert np.max(np.abs(cnew_positions[:, 0] - cold_positions[:, 0])) > 1e-8
