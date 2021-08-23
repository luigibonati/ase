from pathlib import Path
import json
from collections.abc import MutableMapping, Mapping
from contextlib import contextmanager
from ase.io.jsonio import read_json, write_json, encode
from ase.io.ulm import ulmopen
from ase.utils import opencew


def missing(key):
    raise KeyError(key)


class Locked(Exception):
    pass


class CacheLock:
    def __init__(self, fd, key):
        self.fd = fd
        self.key = key

    def save(self, value):
        json_utf8 = encode(value).encode('utf-8')
        try:
            self.fd.write(json_utf8)
        except Exception as ex:
            raise RuntimeError(f'Failed to save {value} to cache') from ex
        finally:
            self.fd.close()


# TODO: This is way hacky.
class CacheLockUlm:
    def __init__(self, fd, key):
        self.fd = fd
        self.key = key

    def save(self, value):
        try:
            self.fd.write("cache", value)
        except Exception as ex:
            raise RuntimeError(f'Failed to save {value} to cache') from ex
        finally:
            self.fd.close()


class JSONBackend:
    extension = '.json'
    MultiFileCache = 'MultiFileJSONCache'
    CombinedCache = 'CombinedJSONCache'
    CL = CacheLock

    @staticmethod
    def open_for_writing(path):
        return opencew(path)

    @staticmethod
    def read(fname):
        return read_json(fname, always_array=False)

    @staticmethod
    def write(target, data):
        write_json(target, data)


class ULMBackend:
    extension = '.ulm'
    MultiFileCache = 'MultiFileULMCache'
    CombinedCache = 'CombinedULMCache'
    CL = CacheLockUlm

    @staticmethod
    def open_for_writing(path):
        return ulmopen(path, 'w')

    @staticmethod
    def read(fname):
        with ulmopen(fname, 'r') as r:
            data = r._data['cache']
        return data

    @staticmethod
    def write(target, data):
        with ulmopen(target, 'w') as w:
            w.write('cache', data)


class _MultiFileCacheTemplate(MutableMapping):
    writable = True
    backend = None

    def __init__(self, directory):
        self.directory = Path(directory)

    def _filename(self, key):
        return self.directory / (f'cache.{key}' + self.backend.extension)

    def _glob(self):
        return self.directory.glob('cache.*' + self.backend.extension)

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
        fd = self.backend.open_for_writing(path)
        try:
            if fd is None:
                yield None
            else:
                yield self.backend.CL(fd, key)
        finally:
            if fd is not None:
                fd.close()

    def __setitem__(self, key, value):
        with self.lock(key) as handle:
            if handle is None:
                raise Locked(key)
            handle.save(value)

    def __getitem__(self, key):
        path = self._filename(key)
        try:
            return self.backend.read(path)
        except FileNotFoundError:
            missing(key)
        except IOError:
            return None
        # Exception for decode error not available for all formats

    def __delitem__(self, key):
        try:
            self._filename(key).unlink()
        except FileNotFoundError:
            missing(key)

    def combine(self):
        cache = globals()[self.backend.CombinedCache].dump_cache(self.directory, dict(self))
        assert set(cache) == set(self)
        self.clear()
        assert len(self) == 0
        return cache

    def split(self):
        return self

    def filecount(self):
        return len(self)

    def strip_empties(self):
        empties = [key for key, value in self.items() if value is None]
        for key in empties:
            del self[key]
        return len(empties)


class _CombinedCacheTemplate(Mapping):
    writable = False
    backend = None

    def __init__(self, directory, dct):
        self.directory = Path(directory)
        self._dct = dict(dct)

    def filecount(self):
        return int(self._filename.is_file())

    @property
    def _filename(self):
        return self.directory / ('combined' + self.backend.extension)

    def __len__(self):
        return len(self._dct)

    def __iter__(self):
        return iter(self._dct)

    def __getitem__(self, index):
        return self._dct[index]

    def _dump(self):
        target = self._filename
        if target.exists():
            raise RuntimeError(f'Already exists: {target}')
        self.directory.mkdir(exist_ok=True, parents=True)
        self.backend.write(target, self._dct)

    @classmethod
    def dump_cache(cls, path, dct):
        cache = cls(path, dct)
        cache._dump()
        return cache

    @classmethod
    def load(cls, path):
        # XXX Very hacky this one
        cache = cls(path, {})
        dct = cls.backend.read(cache._filename)
        cache._dct.update(dct)
        return cache

    def clear(self):
        self._filename.unlink()
        self._dct.clear()

    def combine(self):
        return self

    def split(self):
        cache = globals()[self.backend.MultiFileCache](self.directory)
        assert len(cache) == 0
        cache.update(self)
        assert set(cache) == set(self)
        self.clear()
        return cache


class MultiFileJSONCache(_MultiFileCacheTemplate):
    backend = JSONBackend()

    def __getitem__(self, key):
        path = self._filename(key)
        try:
            return self.backend.read(path)
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


class MultiFileULMCache(_MultiFileCacheTemplate):
    backend = ULMBackend()


class CombinedJSONCache(_CombinedCacheTemplate):
    backend = JSONBackend()


class CombinedULMCache(_CombinedCacheTemplate):
    backend = ULMBackend()


def get_json_cache(directory):
    try:
        return CombinedJSONCache.load(directory)
    except FileNotFoundError:
        return MultiFileJSONCache(directory)
