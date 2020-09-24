import sys

import pytest

from ase.visualize import view
from ase.visualize.external import viewers, PyViewer, CLIViewer
from ase.build import bulk


dummy_args = [sys.executable, '-c', 'print("hello")']


@pytest.fixture
def atoms():
    return bulk('Au')


def test_view_ase(atoms):
    viewer = view(atoms)
    assert viewer.poll() is None
    # Can we stop in a different way?
    viewer.terminate()
    status = viewer.wait()
    assert status != 0


def test_view_ase_via_cli(atoms):
    # Not a very good test, how can we assert something about the viewer?
    viewer = view(atoms, viewer='ase_gui_cli')
    assert viewer.poll() is None
    viewer.terminate()
    status = viewer.wait()
    assert status != 0


def test_bad_viewer(atoms):
    with pytest.raises(KeyError):
        view(atoms, viewer='_nonexistent_viewer')


def test_py_viewer_mock(atoms, monkeypatch):
    def mock_view(self, atoms, repeat=None):
        print(f'viewing {atoms} with mock "{self.name}"')
        return (atoms, self.name)

    monkeypatch.setattr(PyViewer, 'sage', mock_view, raising=False)

    (atoms1, name1) = view(atoms, viewer='sage')
    assert name1 == 'sage'
    assert atoms1 == atoms

    atoms2, name2 = view(atoms, viewer='sage', repeat=(2, 2, 2), block=True)
    assert name2 == 'sage'
    assert len(atoms2) == 8 * len(atoms)


@pytest.mark.parametrize('name', [v.name for v in CLIViewer.viewers()])
def test_cli_viewer_mock(atoms, monkeypatch, name):
    viewer = viewers[name]
    # Can we substitute a more intelligent test than just ase info'ing each?
    monkeypatch.setattr(viewer, 'argv', [sys.executable, '-m', 'ase', 'info'])
    view(atoms, viewer=name, block=True)
