from pathlib import Path
import pytest
import ase
from ase.test.lint import run_flaketest


def have_documentation():
    ase_path = Path(ase.__path__[0])
    doc_path = ase_path.parent / 'doc/ase/ase.rst'
    return doc_path.is_file()


@pytest.mark.slow
@pytest.mark.lint
def test_flake8():
    pytest.importorskip('flake8')
    if not have_documentation():
        pytest.skip('ase/doc not present; '
                    'this is probably an installed version ')

    errmsg = run_flaketest()
    assert errmsg == '', errmsg
