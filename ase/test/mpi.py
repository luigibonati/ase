import threading
from typing import Union, Dict

import numpy as np
from ase.parallel import world


class TestMPIWorldUsingThreads:
    def __init__(self, *, size: int):
        self.size = size
        self.data: Dict[int, np.ndarray] = {}
        self.barrier = threading.Barrier(parties=size, timeout=5.0)

    @property
    def rank(self) -> int:
        return int(threading.current_thread().name)

    def sum(self,
            array: Union[np.ndarray, int, float],
            root: int = None) -> Union[None, int, float]:

        if isinstance(array, (float, int)):
            array1 = np.array([array])
            self.sum(array1, root)
            if root is None or self.rank == root:
                return array1[0]
            return None

        self.data[self.rank] = array

        i = self.barrier.wait()
        if i == 0:
            result = sum(self.data.values())
            for rank, array in self.data.items():
                if root is None or rank == root:
                    array[:] = result
            self.data.clear()

        self.barrier.wait()

        return None

    def broadcast(self, array: np.ndarray, root: int = 0) -> None:
        self.data[self.rank] = array
        i = self.barrier.wait()
        if i == 0:
            for rank, array in self.data.items():
                if rank != root:
                    array[:] = self.data[root]
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


def run_in_parallel(size: int):
    """Decorator ..."""
    def decorator(function):
        def new_function():
            run_function_in_parallel(function, size)
        return new_function
    return decorator
