import io
import ase.io.py
from ase import Atoms

def test_py():
    s=Atoms('H1000')
    with io.StringIO() as buf:
        ase.io.py.write_py(buf, s)
        txt = buf.getvalue()
        assert '...' not in txt
