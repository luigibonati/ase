from pathlib import Path
import json
from collections.abc import MutableMapping, Mapping
from contextlib import contextmanager
from ase.io.jsonio import read_json, write_json, encode
from ase.utils import opencew


def missing(key):
    raise KeyError(key)


class CacheLock:
    def __init__(self, fd):
        self.fd = fd

    def save(self, value):
        json = encode(value)
        self.fd.write(json.encode('utf-8'))


class MultiFileJSONCache(MutableMapping):
    def __init__(self, directory):
        self.directory = Path(directory)

    def _filename(self, key):
        return self.directory / f'cache.{key}.json'

    def _glob(self):
        return self.directory.glob('cache.*.json')

    def __iter__(self):
        for path in self._glob():
            cache, key = path.stem.split('.', 1)
            if cache != 'cache':
                continue
            yield key

    def __len__(self):
        # Very inefficient this, but not a big usecase.
        return len(list(self._glob()))

    @contextmanager
    def lock(self, key):
        self.directory.mkdir(exist_ok=True, parents=True)
        path = self._filename(key)
        fd = opencew(path)
        try:
            if fd is None:
                return None
            else:
                yield CacheLock(fd)
        finally:
            if fd is not None:
                fd.close()

    def __setitem__(self, key, value):
        with self.lock(key) as handle:
            if handle is None:
                raise RuntimeError(f'Locked: {key}')
            handle.save(value)

    def __getitem__(self, key):
        path = self._filename(key)
        try:
            return read_json(path, always_array=False)
        except FileNotFoundError:
            missing(key)
        except json.decoder.JSONDecodeError:
            # May be partially written, which typically means empty
            # because the file was locked with exclusive-write-open.
            #
            # Since we decide what keys we have based on which files exist,
            # we are obligated to return a value for this case too.
            # So we return None.
            return None

    def __delitem__(self, key):
        try:
            self._filename(key).unlink()
        except FileNotFoundError:
            missing(key)

    def combine(self):
        cache = CombinedJSONCache.dump_cache(self.directory, dict(self))
        assert set(cache) == set(self)
        self.clear()
        assert len(self) == 0
        return cache


class CombinedJSONCache(Mapping):
    def __init__(self, directory, dct):
        self.directory = Path(directory)
        self._dct = dict(dct)

    @property
    def _filename(self):
        return self.directory / 'combined.json'

    def _dump_json(self):
        self.directory.mkdir(exist_ok=True, parents=True)
        write_json(self._filename, self._dct)

    def __len__(self):
        return len(self._dct)

    def __iter__(self):
        return iter(self._dct)

    def __getitem__(self, index):
        return self._dct[index]

    @classmethod
    def dump_cache(cls, path, dct):
        cache = cls(path, dct)
        cache._dump_json()
        return cache

    @classmethod
    def load(cls, path):
        dct = read_json(path, always_array=False)
        return cls(path, dct)

    def clear(self):
        self._filename.unlink()
        self._dct.clear()

    def split(self):
        cache = MultiFileJSONCache(self.directory)
        assert len(cache) == 0
        cache.update(self)
        assert set(cache) == set(self)
        self.clear()
        return cache
