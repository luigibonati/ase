import io
import re

import pytest

from ase.build import bulk
from ase.io import write
from ase.io.elk import parse_elk_eigval
from ase.units import Hartree


def test_elk_in():
    atoms = bulk('Si')
    buf = io.StringIO()
    write(buf, atoms, format='elk-in', parameters={'mockparameter': 17})
    text = buf.getvalue()
    print(text)
    assert 'avec' in text
    assert re.search(r'mockparameter\s+17\n', text, re.M)


mock_elk_eigval_out = """
2 : nkpt
3 : nstsv

1   0.0 0.0 0.0 : k-point, vkl
(state, eigenvalue and occupancy below)
1 -1.0 2.0
2 -0.5 1.5
3  1.0 0.0


2   0.0 0.1 0.2 : k-point, vkl
(state, <blah blah>)
1 1.0 1.9
2 1.1 1.8
3 1.2 1.7
"""


def test_parse_eigval():
    fd = io.StringIO(mock_elk_eigval_out)
    dct = dict(parse_elk_eigval(fd))
    eig = dct['eigenvalues'] / Hartree
    occ = dct['occupations']
    kpts = dct['ibz_kpoints']
    assert len(eig) == 1
    assert len(occ) == 1
    assert pytest.approx(eig[0]) == [[-1.0, -0.5, 1.0], [1.0, 1.1, 1.2]]
    assert pytest.approx(occ[0]) == [[2.0, 1.5, 0.0], [1.9, 1.8, 1.7]]
    assert pytest.approx(kpts) == [[0., 0., 0.], [0.0, 0.1, 0.2]]
