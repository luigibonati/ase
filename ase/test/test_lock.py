import pytest


def test_lock():
    """Test timeout on Lock.acquire()."""
    from ase.utils import Lock
    from ase.test import must_raise

    lock = Lock('lockfile', timeout=0.3)
    with lock:
        with must_raise(TimeoutError):
            with lock:
                ...


def test_lock_close_file_descriptor():
    """Test that lock file descriptor is properly closed."""
    from ase.utils import Lock

    lock = Lock('lockfile', timeout=0.3)
    with pytest.warns(None) as record:
        with lock:
            pass

    assert not len(record), record[0].message
