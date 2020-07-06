def test_readwrite_errors():
    import pytest
    from io import StringIO
    from ase.io import read, write
    from ase.build import bulk
    from ase.io.formats import UnknownFileTypeError

    atoms = bulk('Au')
    fd = StringIO()

    with pytest.raises(UnknownFileTypeError):
        write(fd, atoms, format='hello')

    with pytest.raises(UnknownFileTypeError):
        read(fd, format='hello')


def test_parse_filename():
    from ase.io.formats import parse_filename
    filename, index = parse_filename('file_name.traj@1:4:2')
    assert filename == 'file_name.traj'
    assert index == slice(1, 4, 2)
    filename, index = parse_filename('path.to/file@name.traj')
    assert filename == 'path.to/file@name.traj'
    assert index is None
