import pytest
from ase.utils.filecache import MultiFileJSONCache, CombinedJSONCache


@pytest.fixture
def cache():
    return MultiFileJSONCache('cache')


def sample_dict():
    return {'hello': [1, 2, 3], 'world': 'grumble'}


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
    sample_dict(),
])
def test_cache(dct, cache):
    cache.update(dct)
    assert dict(cache) == dct


def test_split(cache):
    dct = sample_dict()
    cache.update(dct)
    combined = cache.combine()
    assert dict(combined) == dct


def test_combine():
    dct = sample_dict()
    combined = CombinedJSONCache.dump_cache('cache', dct)
    assert dict(combined) == dct
    cache = combined.split()
    assert dict(cache) == dct
    assert len(combined) == 0
