import pytest
from ase.utils.filecache import MultiFileJSONCache


@pytest.fixture
def cache():
    return MultiFileJSONCache('cache')


def test_basic(cache):
    assert len(cache) == 0

    cache['hello'] = 'grumble'
    assert len(cache) == 1
    assert 'hello' in cache

    grumble = cache.pop('hello')
    assert grumble == 'grumble'
    assert 'hello' not in cache
    assert len(cache) == 0


@pytest.mark.parametrize('dct', [
    {},
    {'hello': [1, 2, 3], 'world': 'grumble'},
])
def test_cache(dct, cache):
    cache.update(dct)
    assert dict(cache) == dct
