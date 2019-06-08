import itertools
import numpy as np

from ase.geometry.rmsd.cell_projection import intermediate_representation
from ase.geometry.rmsd.standard_form import standardize_atoms
from ase.geometry.rmsd.alignment import LatticeComparator
from ase.geometry.rmsd.lattice_subgroups import get_group_elements


def invert_permutation(p):
    return np.argsort(p)


class LatticeReducer:

    def __init__(self, atoms):

        a = atoms.copy()
        b = atoms.copy()
        res = standardize_atoms(a, b, False)
        atomic_perms, axis_perm = res
        res = intermediate_representation(a, b, 'central', False)
        pa, pb, _, _, _, _ = res

        lc = LatticeComparator(pa, pb)
        dim = lc.dim

        num_atoms = len(lc.numbers)
        nx, ny, nz = len(lc.xindices), len(lc.yindices), len(lc.zindices)
        if dim == 1:
            distances = np.zeros(nz)
            permutations = -np.ones((nz, num_atoms)).astype(np.int)
        elif dim == 2:
            distances = np.zeros((nx, ny))
            permutations = -np.ones((nx, ny, num_atoms)).astype(np.int)
        elif dim == 3:
            distances = np.zeros((nx, ny, nz))
            permutations = -np.ones((nx, ny, nz, num_atoms)).astype(np.int)

        self.lc = lc
        self.cindices = [lc.xindices, lc.yindices, lc.zindices]
        self.distances = distances
        self.permutations = permutations

    def get_point(self, c):

        c = tuple(c)
        if self.permutations[c][0] != -1:
            return self.permutations[c]

        rmsd, permutation = self.lc.cherry_pick(c)
        self.distances[c] = rmsd
        self.permutations[c] = permutation
        return permutation

    def permutationally_consistent(self, H):

        n = len(self.lc.s0)
        dim = self.lc.dim

        seen = -np.ones((3, n)).astype(np.int)
        indices = get_group_elements([n] * dim, H)

        for c in indices:
            p0 = self.get_point(c)
            invp0 = invert_permutation(p0)

            for i, e in enumerate(H):
                c1 = (c + e) % n
                p1 = self.get_point(c1)

                val = p1[invp0]
                if seen[i][0] == -1:
                    seen[i] = val
                elif (seen[i] != val).any():
                    return -float("inf")

        indices = tuple(list(zip(*indices)))
        return np.sqrt(np.sum(self.distances[indices]**2))
