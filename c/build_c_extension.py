from cffi import FFI
ffibuilder = FFI()

ffibuilder.cdef("double dos(double* a, int n);")

ffibuilder.set_source("_ase",  # name of the output C extension
"""
double dos(double* a, int n);
""",
    #include "pi.h"',
    sources=['c/dos.c'],   # includes pi.c as additional sources
    libraries=[])    # on Unix, link with the math library

if __name__ == "__main__":
    ffibuilder.compile(verbose=True)
    