import pytest
from ase.atoms import Atoms
from ase.optimize.precon import PreconLBFGS


def test_precon_warn():
    with pytest.warns(UserWarning, match='The system is likely too small'):
        PreconLBFGS(Atoms('H'))


@pytest.mark.skip('FAILS WITH PYAMG')
@pytest.mark.slow
@pytest.mark.parametrize('N', [1, 3])
def test_precon(N):
    a0 = bulk('Cu', cubic=True)
    a0 *= (N, N, N)

def test_precon_nowarn():
    PreconLBFGS(Atoms('100H'))
