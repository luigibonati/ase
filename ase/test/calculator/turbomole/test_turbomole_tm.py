# type: ignore
import pytest
from ase.calculators.turbomole import Turbomole


@pytest.fixture
def default_params():
    return {'multiplicity': 1}


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
