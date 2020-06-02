import pytest
import numpy as np
from ase.calculators.calculator import equal


def generate(a, dtype):
    return [list(map(dtype, a)), np.array(a, dtype=dtype)]


@pytest.mark.parametrize('a', generate([1, 1], int) + generate([1, 1], float))
@pytest.mark.parametrize('b', generate([1, 1], int) + generate([1, 1], float))
@pytest.mark.parametrize('rtol', [None, 0, 1e-8])
@pytest.mark.parametrize('atol', [None, 0, 1e-8])
def test_equal(a, b, rtol, atol):
    assert a is not b
    assert equal(a, b, rtol=rtol, atol=atol)
    assert equal({'size': a, 'gamma': True},
                 {'size': b, 'gamma': True}, rtol=rtol, atol=atol)
    assert not equal({'size': a, 'gamma': True},
                     {'size': b, 'gamma': False}, rtol=rtol, atol=atol)


@pytest.mark.parametrize('a', generate([2, 2], float))
@pytest.mark.parametrize('b', generate(np.array([2, 2]) + 1.9e-8, float))
@pytest.mark.parametrize('rtol,atol', [[None, 2e-8], [0, 2e-8],
                                       [1e-8, None], [1e-8, 0],
                                       [0.5e-8, 1e-8]])
def test_almost_equal(a, b, rtol, atol):
    assert a is not b
    assert equal(a, b, rtol=rtol, atol=atol)
    assert equal({'size': a, 'gamma': True},
                 {'size': b, 'gamma': True}, rtol=rtol, atol=atol)


@pytest.mark.parametrize('a', generate([2, 2], float))
@pytest.mark.parametrize('b', generate(np.array([2, 2]) + 3.1e-8, float))
@pytest.mark.parametrize('rtol', [None, 0, 1e-8])
@pytest.mark.parametrize('atol', [None, 0, 1e-8])
def test_not_equal(a, b, rtol, atol):
    assert a is not b
    assert not equal(a, b, rtol=rtol, atol=atol)
    assert not equal({'size': a, 'gamma': True},
                     {'size': b, 'gamma': True}, rtol=rtol, atol=atol)
