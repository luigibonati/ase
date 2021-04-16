from io import StringIO
import re
import pytest
from ase.io.aims import write_aims_control, value2string
from ase.build import bulk


def control2txt(**parameters):
    buf = StringIO()
    write_aims_control(buf, bulk('Au'), parameters)
    return buf.getvalue()


def test_empty():
    txt = control2txt()
    assert 'FHI-aims control file' in txt


def test_kpts():
    txt = control2txt(kpts=[4, 4, 4])
    assert re.search(r'k_grid\s+4 4 4', txt)


def test_smearing():
    txt = control2txt(smearing=('bananas', 17))
    assert re.search(r'occupation_type\s+bananas\s+17', txt)


@pytest.mark.parametrize('value, string', [
    (True, '.true.'),
    (False, '.false.'),
    (None, ''),
    ('potato', 'potato'),
    (0.123, '0.123'),
    (42, '42'),
    ([2, 3, 4], '2 3 4'),
])
def test_value2string(value, string):
    assert value2string(value) == string
