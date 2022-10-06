import pytest
from ase.io.formats import ioformats


def test_manually():
    traj = ioformats['traj']
    print(traj)

    outcar = ioformats['vasp-out']
    print(outcar)
    assert outcar.match_name('OUTCAR')
    assert outcar.match_name('something.with.OUTCAR.stuff')


@pytest.mark.parametrize('name', ioformats)
def test_ioformat(name):
    """Test getting the full description of each ioformat."""
    if name == 'exciting':
        # Check if excitingtools is installed, if not skip exciting tests.
        try:
            __import__('excitingtools')
        except ModuleNotFoundError:
            pytest.skip(
                'excitingtools not installed so skipping exciting test.')
    ioformat = ioformats[name]
    print(name)
    print('=' * len(name))
    print(ioformat.full_description())
    print()
