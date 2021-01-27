"""Testing of "ase db" command-line interface."""
from pathlib import Path

import pytest
from ase import Atoms
from ase.db import connect


@pytest.fixture(scope='module')
def dbfile(tmp_path_factory) -> Path:
    """Create a database file (x.db) with two rows."""
    path = tmp_path_factory.mktemp('db') / 'x.db'

    with connect(path) as db:
        db.write(Atoms())
        db.write(Atoms())

    return path


def test_insert_into(cli, dbfile):
    """Test --insert-into."""
    out = dbfile.with_name('x1.db')
    # Insert 1 row:
    cli.ase(
        *f'db {dbfile} --limit 1 --insert-into {out} --progress-bar'.split())
    # Count:
    txt = cli.ase(*f'db {out} --count'.split())
    num = int(txt.split()[0])
    assert num == 1
