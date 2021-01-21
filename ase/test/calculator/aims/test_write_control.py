from io import StringIO
import re
from ase.io.aims import write_aims_control
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
    assert re.search(r'k_offset\s+0\.37', txt)  # What's with the offset?


def test_smearing():
    txt = control2txt(smearing=('bananas', 17))
    assert re.search(r'occupation_type\s+bananas\s+17', txt)
