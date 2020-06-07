import pytest
from ase.utils import Lock


def test_lock_acquire_timeout():
    lock = Lock('lockfile', timeout=0.3)
    with lock:
        with pytest.raises(TimeoutError):
            with lock:
                ...
