import pytest
from io import BytesIO
from ase.io.formats import ioformats, filetype
from ase.io import write


def lammpsdump_headers():
    actual_magic = 'ITEM: TIMESTEP'
    yield actual_magic
    yield f'anything\n{actual_magic}\nanything'


from fnmatch import fnmatchcase


@pytest.mark.parametrize('header', lammpsdump_headers())
def test_recognize_file(header):
    fmt_name = 'lammps-dump-text'
    fmt = ioformats[fmt_name]
    assert fmt.match_magic(header.encode('ascii'))
