import numpy as np


def dos(a: np.ndarray):
    try:
        from _ase import lib, ffi
        return lib.dos(ffi.from_buffer('double[]', a), len(a))
    except ImportError:
        pass
    print('Python')
    return a[-1]


a = np.array([0.0, 42.0])
x = dos(a)
print(x)
