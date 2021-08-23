import numpy as np
import pytest
from ase.utils.filecache import (MultiFileJSONCache, CombinedJSONCache,
                                 MultiFileULMCache,
                                 Locked)


pytestmark = pytest.mark.usefixtures('testdir')


cache_types = ['json', 'ulm']


@pytest.fixture
def caches(cache_type=None):
    return {'json': MultiFileJSONCache('cache_json'),
            'ulm': MultiFileULMCache('cache_ulm')}


def sample_dict():
    return {'hello': [1, 2, 3], 'world': 'grumble'}


@pytest.mark.parametrize('cache_type', cache_types)
def test_basic(caches, cache_type):
    cache = caches[cache_type]
    assert len(cache) == 0

    cache['hello'] = 'grumble'
    assert len(cache) == 1
    assert 'hello' in cache

    grumble = cache.pop('hello')
    assert grumble == 'grumble'
    assert 'hello' not in cache
    assert len(cache) == 0


@pytest.mark.parametrize('cache_type', cache_types)
def test_numpy_array(caches, cache_type):
    cache = caches[cache_type]
    assert len(cache) == 0
    cache['a'] = np.array([3.4, 2.4, 1.4j])
    cache['b'] = np.array([3.4, 2.4, 1.4])
    assert len(cache) == 2
    assert 'a' in cache
    aa = cache.pop('a')
    bb = cache.pop('b')
    assert np.allclose(aa, np.array([3.4, 2.4, 1.4j]))
    assert np.allclose(bb, np.array([3.4, 2.4, 1.4]))
    assert len(cache) == 0


@pytest.mark.parametrize('dct', [{}, sample_dict()])
@pytest.mark.parametrize('cache_type', cache_types)
def test_cache(dct, caches, cache_type):
    cache = caches[cache_type]
    cache.update(dct)
    assert dict(cache) == dct


@pytest.mark.parametrize('cache_type', cache_types)
def test_combine(caches, cache_type):
    cache = caches[cache_type]
    dct = sample_dict()
    cache.update(dct)
    combined = cache.combine()
    assert dict(combined) == dct


def test_split():
    dct = sample_dict()
    combined = CombinedJSONCache.dump_cache('cache', dct)
    assert dict(combined) == dct
    cache = combined.split()
    assert dict(cache) == dct
    assert len(combined) == 0


@pytest.mark.parametrize('cache_type', cache_types)
def test_lock(caches, cache_type):
    cache = caches[cache_type]
    with cache.lock('hello'):
        # When element is locked but nothing is written, the
        # cache is defined to "contain" None
        assert 'hello' in cache
        assert cache['hello'] is None

        # Other keys should function as normal:
        cache['xx'] = 1
        assert cache['xx'] == 1


@pytest.mark.parametrize('cache_type', cache_types)
def test_already_locked(caches, cache_type):
    cache = caches[cache_type]
    with cache.lock('hello') as handle:
        assert handle is not None
        with cache.lock('hello') as otherhandle:
            assert otherhandle is None

        with pytest.raises(Locked):
            cache['hello'] = 'world'


@pytest.mark.parametrize('cache_type', cache_types)
def test_no_overwrite_combine(caches, cache_type):
    cache = caches[cache_type]
    cache.combine()
    with pytest.raises(RuntimeError, match='Already exists'):
        cache.combine()
