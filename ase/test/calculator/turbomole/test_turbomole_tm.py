"""unit tests for turbomole calculator that need no turbomole installed"""
# type: ignore
import pytest
import numpy as np
from ase.calculators.turbomole import Turbomole

@pytest.fixture
def default_params():
    return {'multiplicity': 1}

#@pytest.fixture
#def calc(params):
#    return Turbomole(**params)

def test_turbomole_empty():
    with pytest.raises(AssertionError) as err:
        assert Turbomole()
        assert str(err.value) == 'multiplicity not defined'

def test_turbomole_default(default_params):
    calc = Turbomole(**default_params)
    assert calc['label'] is None
    assert calc['prefix'] is None
    assert calc['directory'] == '.'
    assert not calc['restart']
    assert calc['atoms'] is None
#    assert calc['define_str'] is None
#
#    assert not calc['restart']
