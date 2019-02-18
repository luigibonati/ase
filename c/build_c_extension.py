from pathlib import Path
from cffi import FFI
ffibuilder = FFI()

ffibuilder.cdef(Path('_ase.h').read_text())

ffibuilder.set_source('_ase',
                      '#include "_ase.h"\n',
                      sources=[path for path in Path().glob('*.c')
                               if path.name != '_ase.c'])

ffibuilder.emit_c_code('_ase.c')
