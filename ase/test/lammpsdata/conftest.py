import pytest

@pytest.fixture
def lammpsdata_file_path(datadir):
    return datadir / "lammpsdata_input.data"
