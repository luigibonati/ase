import pytest
from ase.atoms import Atoms
from ase.build import bulk
from ase.optimize.precon import PreconLBFGS
from ase.calculators.emt import EMT

def test_precon_warn():
    with pytest.warns(UserWarning, match='The system is likely too small'):
        PreconLBFGS(Atoms('H'))


@pytest.mark.skip('FAILS WITH PYAMG')
@pytest.mark.slow
@pytest.mark.parametrize('N', [1, 3])
def test_precon(N):
    atoms = bulk('Cu', cubic=True)
    atoms = (N, N, N)
    atoms.calc = EMT()
    opt = PreconLBFGS(atoms, precon="auto")

def test_precon_nowarn():
    PreconLBFGS(Atoms('100H'))
