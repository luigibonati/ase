import pytest
import numpy as np
from ase.spacegroup import Spacegroup
from ase.lattice import FCC


def test_spacegroup_miscellaneous():
    no = 225
    sg = Spacegroup(no)
    assert int(sg) == no == sg.no
    assert sg.centrosymmetric
    assert sg.symbol == 'F m -3 m'
    assert sg.symbol in str(sg)
    assert sg.lattice == 'F'  # face-centered
    assert sg.setting == 1
    assert sg.scaled_primitive_cell == pytest.approx(FCC(1.0).tocell()[:])
    assert sg.reciprocal_cell @ sg.scaled_primitive_cell == pytest.approx(
        np.identity(3))
