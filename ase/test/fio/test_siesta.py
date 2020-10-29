from io import StringIO
import numpy as np
import pytest
from ase.io.siesta import read_struct_out

sample_struct_out = """\
  3.0   0.0   0.0
 -1.5   4.0   0.0
  0.0   0.0   5.0
        2
   1   45   0.0   0.0   0.0
   1   46   0.3   0.4   0.5
"""

def test_read_struct_out():
    atoms = read_struct_out(StringIO(sample_struct_out))
    assert all(atoms.numbers == [45, 46])
    assert atoms.get_scaled_positions() == pytest.approx(
        np.array([[0., 0., 0.], [.3, .4, .5]]))
    assert atoms.cell[:] == pytest.approx(np.array([[3.0, 0.0, 0.0],
                                                    [-1.5, 4.0, 0.0],
                                                    [0.0, 0.0, 5.0]]))
    assert all(atoms.pbc)
