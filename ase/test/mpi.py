import threading

import numpy as np
from ase.parallel import world


class TestMPIWorldUsingThreads:
    def __init__(self, *, size: int):
        self.size = size
        self.data = {}
        self.barrier = threading.Barrier(parties=size, timeout=5.0)

    @property
    def rank(self) -> int:
        return int(threading.current_thread().name)

    def sum(self, array, rank: int = None):
        if isinstance(array, (float, int)):
            a = np.array([array])
            self.sum(a, rank)
            if rank is None or self.rank == rank:
                return a[0]
            return None
        self.data[self.rank] = array
        i = self.barrier.wait()
        if i == 0:
            result = sum(self.data.values())
            for r, a in self.data.items():
                if rank is None or r == rank:
                    a[:] = result
            self.data.clear()
        self.barrier.wait()

    def broadcast(self, array: np.ndarray, rank: int = 0) -> None:
        self.data[self.rank] = array
        i = self.barrier.wait()
        if i == 0:
            for r, a in self.data.items():
                if r != rank:
                    a[:] = self.data[rank]
            self.data.clear()
        self.barrier.wait()


def run_function_in_parallel(function, size):
    testworld = TestMPIWorldUsingThreads(size=size)
    try:
        world.comm = testworld
        threads = []
        for rank in range(size):
            thread = threading.Thread(target=function, name=str(rank))
            thread.start()
            threads.append(thread)
        for thread in threads:
            thread.join()
    finally:
        world.comm = None


def run_in_parallel(size):
    """Decorator ..."""
    def decorator(function):
        def new_function():
            run_function_in_parallel(function, size)
        return new_function
    return decorator
