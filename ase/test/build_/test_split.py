from ase.build import molecule
from ase.build.split import split_bond


def test_split_biphenyl():
    mol = molecule('biphenyl')

    mol1, mol2 = split_bond(mol, 0, 14)
    assert len(mol) == len(mol1) + len(mol2)
    mol2s, mol1s = split_bond(mol, 14, 0)
    assert mol1s == mol1
    assert mol2s == mol2

    # we cannot split within the ring
    mol1, mol2 = split_bond(mol, 0, 1)
    assert len(mol) < len(mol1) + len(mol2)
