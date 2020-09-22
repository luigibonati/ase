import filecmp
from ase.build import molecule
from ase.io import write
from ase.calculators.emt import EMT


def test_single_write():
    atoms = molecule('H2O')
    write('1.xyz', atoms, format='extxyz', plain=True)
    write('2.xyz', atoms, format='xyz', fmt='%16.8f')
    assert filecmp.cmp('1.xyz', '2.xyz', shallow=False), 'Files differ'


def test_single_write_with_forces():
    # Create atoms with forces and test that
    # those aren't written to the file
    atoms = molecule('CO')
    atoms.calc = EMT()
    atoms.get_forces()
    write('1.xyz', atoms, format='extxyz', plain=True)
    write('2.xyz', atoms, format='xyz', fmt='%16.8f')
    write('3.xyz', molecule('CO'), format='xyz', fmt='%16.8f')
    assert filecmp.cmp('1.xyz', '2.xyz', shallow=False), 'Files differ'
    assert filecmp.cmp('1.xyz', '3.xyz', shallow=False), 'Files differ'


def test_single_write_with_comment():
    atoms = molecule('H2O')
    comment = 'my comment'
    write('1.xyz', atoms, format='extxyz', plain=True, comment=comment)
    write('2.xyz', atoms, format='xyz', fmt='%16.8f', comment=comment)
    assert filecmp.cmp('1.xyz', '2.xyz', shallow=False), 'Files differ'


def test_multiple_write():
    images = []
    for name in ['C6H6', 'H2O', 'CO']:
        images.append(molecule(name))
    write('1.xyz', images, format='extxyz', plain=True)
    write('2.xyz', images, format='xyz', fmt='%16.8f')
    assert filecmp.cmp('1.xyz', '2.xyz', shallow=False), 'Files differ'
