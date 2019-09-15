"""Test the Tags class.

The purpose is to verify that all the indexing and similar mechanisms
works correctly."""

import numpy as np
from ase.build import molecule

atoms = molecule('CH3CH2OH')
natoms = len(atoms)

assert not atoms.tags
assert len(atoms.tags) == natoms
assert len(atoms.tags[:3]) == 3

assert all(atoms.tags == 0)
assert list(atoms.tags) == [0] * natoms
assert list(atoms.tags == 0) == [True] * natoms
assert atoms.tags[0] == 0
assert all(atoms.tags[:3] == (0, 0, 0))

tag = 17
atoms.tags[:3] = tag
assert atoms.tags
assert list(atoms.tags[:3]) == 3 * [tag]
assert sum(atoms.tags) == 3 * tag


newtags = np.arange(len(atoms)) // 2
atoms.tags[:] = newtags
assert (atoms.tags == newtags).all()
atoms.tags = 0
assert not atoms.tags
atoms.tags = newtags
assert atoms.tags
assert (atoms.tags == newtags).all()

atoms.tags = 0
assert not any(atoms.tags)

atoms.tags = 1
assert not any(atoms.tags - 1)

print(atoms.tags)

atoms.tags = range(len(atoms))
assert sum(atoms.tags) == natoms * (natoms - 1) // 2

atoms.tags = 0
atoms.tags += 1
assert not any(atoms.tags - 1)

#atoms.tags[3:5] += 17

#assert all(atoms.tags == range(1, 1 + len(atoms)))

tagsum = sum(atoms.tags)
atoms.tags *= 2
assert sum(atoms.tags) == 2 * tagsum
print(atoms.tags + 2)

atoms.tags = 0
sometags = atoms.tags[::2]
sometags[:3] += 1
print(atoms.tags)
assert sum(atoms.tags) == 3
