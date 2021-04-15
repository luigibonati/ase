import os
import re
import warnings

import pytest
import numpy as np
import ase
import ase.lattice.cubic
from ase.calculators.castep import (Castep, CastepOption,
                                    CastepParam, CastepCell,
                                    make_cell_dict, make_param_dict,
                                    CastepKeywords,
                                    create_castep_keywords,
                                    import_castep_keywords,
                                    CastepVersionError)

from ase.test.calculator.castep.fake_keywords import (fake_keywords_data,
                                                      fake_keywords_types,
                                                      fake_keywords_levels)

calc = pytest.mark.calculator

has_castep = (os.environ.get('CASTEP_COMMAND', None) is not None)


@pytest.fixture
def castep_command():
    return os.environ['CASTEP_COMMAND']


@pytest.fixture
def fake_castep_keywords():
    # Generate fake CastepKeywords to use when the real ones aren't necessary
    pdict = make_param_dict(fake_keywords_data)
    cdict = make_cell_dict(fake_keywords_data)
    return CastepKeywords(pdict, cdict,
                          fake_keywords_types,
                          fake_keywords_levels,
                          'False Castep')


@pytest.fixture
def castep_keywords(castep_command, tmp_path):
    create_castep_keywords(castep_command=castep_command, path=tmp_path,
                           fetch_only=20)
    with pytest.warns(None):
        castep_keywords = import_castep_keywords(castep_command=castep_command,
                                                 path=tmp_path)
    return castep_keywords


@pytest.mark.skipif(not has_castep, reason='No Castep Installed')
def test(castep_keywords):
    pass


def test_fundamental_params():
    # Start by testing the fundamental parts of a CastepCell/CastepParam object
    boolOpt = CastepOption('test_bool', 'basic', 'defined')
    boolOpt.value = 'TRUE'
    assert boolOpt.raw_value is True

    float3Opt = CastepOption('test_float3', 'basic', 'real vector')
    float3Opt.value = '1.0 2.0 3.0'
    assert np.isclose(float3Opt.raw_value, [1, 2, 3]).all()

    # Generate a mock keywords object
    mock_castep_keywords = CastepKeywords(make_param_dict(), make_cell_dict(),
                                          [], [], 0)
    mock_cparam = CastepParam(mock_castep_keywords, keyword_tolerance=2)
    mock_ccell = CastepCell(mock_castep_keywords, keyword_tolerance=2)

    # Test special parsers
    mock_cparam.continuation = 'default'
    with pytest.warns(None):
        mock_cparam.reuse = 'default'
    assert mock_cparam.reuse.value is None

    mock_ccell.species_pot = ('Si', 'Si.usp')
    mock_ccell.species_pot = ('C', 'C.usp')
    assert 'Si Si.usp' in mock_ccell.species_pot.value
    assert 'C C.usp' in mock_ccell.species_pot.value
    symops = (np.eye(3)[None], np.zeros(3)[None])
    mock_ccell.symmetry_ops = symops
    assert """1.0 0.0 0.0
0.0 1.0 0.0
0.0 0.0 1.0
0.0 0.0 0.0""" == mock_ccell.symmetry_ops.value.strip()


def test_castep_option(fake_castep_keywords):

    # check if the CastepOption assignment and comparison mechanisms work
    p1 = CastepParam(fake_castep_keywords)
    p2 = CastepParam(fake_castep_keywords)

    assert p1._options == p2._options

    # Set some values
    p1.fake_real_kw = 3.0
    p1.fake_string_kw = 'PBE'
    p1.fake_defined_kw = True
    p1.fake_integer_kw = 10
    p1.fake_integer_vector_kw = [3,3,3]
    p1.fake_real_vector_kw = [3.0,3.0,3.0]
    p1.fake_boolean_kw = False
    p1.fake_physical_kw = '3.0 ang'

    assert p1.fake_real_kw.value == '3.0'
    assert p1.fake_string_kw.value == 'PBE'
    assert p1.fake_defined_kw.value == 'TRUE'
    assert p1.fake_integer_kw.value == '10'
    assert p1.fake_integer_vector_kw.value == '3 3 3'
    assert p1.fake_real_vector_kw.value == '3.0 3.0 3.0'
    assert p1.fake_boolean_kw.value == 'FALSE'
    assert p1.fake_physical_kw.value == '3.0 ang'

    assert p1._options != p2._options

