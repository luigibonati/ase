import numpy as np
from ase.build import bulk, mx2, nanotube
from ase.geometry.rmsd import find_crystal_reductions


TOL = 1E-10


def permute_atoms(atoms, perm):
    atoms = atoms.copy()
    atoms.set_positions(atoms.get_positions()[:, perm])
    atoms.set_cell(atoms.cell[perm][:, perm], scale_atoms=False)
    atoms.set_pbc(atoms.pbc[perm])
    return atoms

# 3-dimensional: NaCl
size = 3
atoms = bulk("NaCl", "rocksalt", a=5.64) * size
result = find_crystal_reductions(atoms)
assert len(result) == size
assert all([reduced.rmsd < TOL for reduced in result])
factors = [reduced.factor for reduced in result]
assert tuple(factors) == (3, 9, 27)

# 2-dimensional: MoS2
for i in range(3):
    size = 4
    atoms = mx2(formula='MoS2', size=(size, size, 1))
    permutation = np.roll(np.arange(3), i)
    atoms = permute_atoms(atoms, permutation)

    result = find_crystal_reductions(atoms)
    assert len(result) == size
    assert all([reduced.rmsd < TOL for reduced in result])
    factors = [reduced.factor for reduced in result]
    assert tuple(factors) == (2, 4, 8, 16)

# 1-dimensional: carbon nanotube
for i in range(3):
    size = 4
    atoms = nanotube(3, 3, length=size)
    permutation = np.roll(np.arange(3), i)
    atoms = permute_atoms(atoms, permutation)

    result = find_crystal_reductions(atoms)
    factors = [reduced.factor for reduced in result[:2]]
    assert tuple(factors) == (2, 4)
    assert all([reduced.rmsd < TOL for reduced in result[:2]])
