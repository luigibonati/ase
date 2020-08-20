import pytest
from ase.io.bytes import to_bytes
from ase.build import bulk


from ase.io.cif import CIFLoop, parse_loop


def test_cifloop():
    dct = {'_eggs': range(4),
           '_potatoes': [1.3, 7.1, -1, 0]}

    loop = CIFLoop()
    loop.add('_eggs', '{:<2d}', dct['_eggs'])
    loop.add('_potatoes', '{:.4f}', dct['_potatoes'])

    string = loop.tostring() + '\n\n'
    lines = string.splitlines()[::-1]
    assert lines.pop() == 'loop_'

    newdct = parse_loop(lines)
    print(newdct)
    assert set(dct) == set(newdct)
    for name in dct:
        assert dct[name] == pytest.approx(newdct[name])
