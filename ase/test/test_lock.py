import pytest
import os


def test_cannot_acquire_lock_twice():
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

    # The choice of timeout=1.0 is arbitrary but we don't want to use
    # something that is too large since it could mean that the test
    # takes long to fail.
    lock = Lock('lockfile', timeout=1.0)
    with lock:
        pass

    # If fstat raises OSError this means that fd.close() was
    # not called.
    with pytest.raises(OSError):
        os.fstat(lock.fd.name)
