import pytest
from ase.io.cif import CIFBlock


@pytest.fixture
def cifblock():
    return CIFBlock('hello', {'_cifkey': 42})


def test_repr(cifblock):
    text = repr(cifblock)
    assert 'hello' in text
    assert '_cifkey' in text


def test_mapping(cifblock):
    assert len(cifblock) == 1
    assert len(list(cifblock)) == 1


def test_various(cifblock):
    assert cifblock.get_cellpar() is None
    assert cifblock.get_cell().rank == 0


def test_deuterium():
    # Verify that the symbol 'D' becomes hydrogen ('H') with mass 2(-ish).
    symbols = ['H', 'D', 'D', 'He']
    block = CIFBlock('deuterium', dict(_atom_site_type_symbol=symbols))
    assert block.get_symbols() == ['H', 'H', 'H', 'He']
    masses = block._get_masses()
    assert all(masses.round().astype(int) == [1, 2, 2, 4])
