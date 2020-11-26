import pytest
from ase.atoms import Atoms
from ase.build import bulk
from ase.optimize.precon import PreconLBFGS
from ase.calculators.emt import EMT

def test_precon_warn():
    with pytest.warns(UserWarning, match='The system is likely too small'):
        PreconLBFGS(Atoms('H'))


def test_precon_nowarn():
    PreconLBFGS(Atoms('100H'))
