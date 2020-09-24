import sys
from pathlib import Path

import pytest

from ase.visualize import view
from ase.visualize.external import viewers, PyViewer, CLIViewer
#from ase.visualize.external import open_external_viewer
from ase.build import bulk

dummy_args = [sys.executable, '-c', 'print("hello")']


#def test_file_missing():
    #viewer = open_external_viewer(dummy_args, 'file_which_does_not_exist')
    #status = viewer.wait()
    #assert status != 0


def test_ok_including_cleanup():
    atoms = bulk('Au')
    viewer = view(atoms, 'ase_gui_cli')
    print(viewer)
    viewer.terminate()
    status = viewer.wait()
    assert status != 0
    #temppath = Path('tmp.txt')
    #temppath.write_text('hello')
    #assert temppath.exists()
    #viewer = open_external_viewer(dummy_args, str(temppath))
    #status = viewer.wait()
    #assert status == 0
    #assert not temppath.exists()


def test_view_ase_gui():
    from ase.build import bulk
    atoms = bulk('Au')
    viewer = view(atoms)
    assert viewer.poll() is None
    # Can we stop in a different way?
    viewer.terminate()
    status = viewer.wait()
    assert status != 0


@pytest.fixture
def atoms():
    return bulk('Au')


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


def mock_execute(self, filename):
    print(f'mock "{self.name} argv={self.argv} on "{filename}"')
    assert self.name == name
    path = Path(filename)
    assert path.exists()
    assert path.suffix == viewer.ioformat.extensions[0]


@pytest.mark.parametrize('name', [v.name for v in CLIViewer.viewers()])
def test_cli_viewer_mock(atoms, monkeypatch, name):
    viewer = viewers[name]
    # Can we substitute a more intelligent test than just ase info'ing each?
    monkeypatch.setattr(viewer, 'argv', [sys.executable, '-m', 'ase', 'info'])
    view(atoms, viewer=name, block=True)
