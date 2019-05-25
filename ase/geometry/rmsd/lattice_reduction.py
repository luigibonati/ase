import numpy as np
from collections import namedtuple
from ase.geometry.rmsd.cell_projection import intermediate_representation
from ase.geometry.rmsd.alignment import best_alignment
from ase.geometry.rmsd.standard_form import standardize_atoms


def invert_permutation(perm):

    n = len(perm)
    invperm = np.arange(n)
    for i, e in enumerate(perm):
        invperm[e] = i
    return invperm


def find_lattice_reductions(atoms):

    res = standardize_atoms(atoms.copy(), atoms.copy(), False)
    a, b, atomic_perms, axis_perm = res

    pa, pb, celldist, mr_path = intermediate_representation(a, b, 'central')
    res = best_alignment(pa, pb, False, None, True)
    rmsd, assignment, U, translation, distances, permutations = res
    return distances, permutations

    # undo Minkowski-reduction
    inv_mr = np.linalg.inv(mr_path)
    imcell = np.dot(inv_mr, pa.cell)

    # convert basis and translation to fractional coordinates
    fractional = np.linalg.solve(imcell.T, translation.T).T
    basis = np.dot(imcell, np.linalg.solve(imcell.T, U.T).T)
    basis *= sign

    # undo atomic permutation
    left, right = atomic_perms
    invleft = invert_permutation(left)
    assignment = invert_permutation(right[assignment[invleft]])

    # undo axis permutation
    invaxis = invert_permutation(axis_perm)
    imcell = imcell[invaxis][:, invaxis]
    fractional = fractional[invaxis]
    basis = basis[invaxis][:, invaxis]

    entries = 'rmsd dcell cell basis translation permutation mul1 mul2'
    result = namedtuple('RMSDResult', entries)
    return result(rmsd=rmsd, dcell=celldist, cell=imcell, basis=basis,
                  translation=fractional, permutation=assignment,
                  mul1=multiplier1, mul2=multiplier2)
