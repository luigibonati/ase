import pytest
from ase.io.bytes import to_bytes
from ase.build import bulk


from ase.io.cif import CIFLoop, parse_loop


def test_cifloop():
    dct = {'_eggs': range(4),
           '_potatoes': [1.3, 7.1, -1, 0]}

    loop = CIFLoop()
    loop.add('_eggs', dct['_eggs'], '{:<2d}')
    loop.add('_potatoes', dct['_potatoes'], '{:.4f}')

    string = loop.tostring() + '\n\n'
    lines = string.splitlines()[::-1]
    assert lines.pop() == 'loop_'

    newdct = parse_loop(lines)
    print(newdct)
    assert set(dct) == set(newdct)
    for name in dct:
        assert dct[name] == pytest.approx(newdct[name])


def test_cif_file():
    atoms = bulk('Ti')
    b = to_bytes(atoms, format='cif')
    txt = b.decode('latin1')
    print('--------')
    print(txt)

    print('--------')
    loop = bloody_loop()
    print(loop.tostring())
    print('--------')


def bloody_loop():
    atoms = bulk('Ti')
    occupancies = [1] * len(atoms)
    coords = atoms.get_scaled_positions()

    loop = CIFLoop()
    loop.add('_atom_site_label', atoms.symbols, '{:<8s}')
    loop.add('_atom_site_occupancy', occupancies, '{:6.4f}')

    for i, axisname in enumerate('xyz'):
        loop.add(f'_atom_site_fract_{axisname}',
                 coords[:, i], '{:7.5f}')

    loop.add('_atom_site_thermal_displace_type',
             ['Biso'] * len(atoms), '{:s}')
    loop.add('_atom_site_B_iso_or_equiv', [1.0] * len(atoms), '{:6.3f}')
    loop.add('_atom_site_type_symbol', atoms.symbols, '{:<2s}')
    return loop
