import pytest
import numpy as np
from ase.dft.kpoints import resolve_custom_points


@pytest.fixture
def special_points():
    return dict(A=np.zeros(3),
                B=np.ones(3))


def test_str(special_points):
    path, dct = resolve_custom_points('AB', special_points, 0)
    assert path == 'AB'
    assert set(dct) == set('AB')


def test_recognize_points_from_coords(special_points):
    path, dct = resolve_custom_points(
        [[special_points['A'], special_points['B']]], special_points, 1e-5)
    assert path == 'AB'
    assert set(dct) == set('AB')


def test_autolabel_points_from_coords(special_points):
    path, dct = resolve_custom_points(
        [[special_points['A'], special_points['B']]], {}, 0)

    assert path == 'Kpt0Kpt1'
    assert set(dct) == {'Kpt0', 'Kpt1'}  # automatically labelled


@pytest.mark.parametrize('bad_pathspec', [
    [np.zeros(3), np.ones(3)],  # Missing one level of nesting
    [[np.zeros(2)]],
])
def test_bad_args(bad_pathspec):
    # BZ paths must be list of list of kpoint
    with pytest.raises(ValueError):
        resolve_custom_points(bad_pathspec, {}, 0)
