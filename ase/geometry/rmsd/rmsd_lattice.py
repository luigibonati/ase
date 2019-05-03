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


def _calculate_rmsd(atoms0, atoms1, frame, ignore_stoichiometry, sign,
                    allow_rotation, num_chain_steps=None):

    
    res = standardize_atoms(atoms0.copy(), atoms1.copy(), ignore_stoichiometry)
    a, b, atomic_perms, axis_perm = res

    pa, pb, celldist, mr_path = intermediate_representation(a, b, frame)
    rmsd, assignment, U, translation = best_alignment(pa, pb, allow_rotation,
                                                      num_chain_steps)

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

    result = namedtuple('RMSDResult',
                        'rmsd dcell cell basis translation permutation')
    return result(rmsd=rmsd, dcell=celldist, cell=imcell, basis=basis,
                  translation=fractional, permutation=assignment)


def calculate_rmsd(atoms1, atoms2, frame='central', ignore_stoichiometry=False,
                   allow_reflection=False, allow_rotation=False):
    """Calculates the optimal RMSD between two crystalline structures.

    A common frame of reference is calculated and the optimal translation is
    found.  The periodic boundary conditions of the input atoms objects
    determine the type of comparison to be made (along a line, in-plane, or a
    full three-dimensional).  Non-crystalline (or molecular) comparisons, i.e.
    comparisons with no periodicity, are not supported.  For 1D (chain-like)
    structures the optimal rotation about the chain axis can optionally be
    found.  Note that the comparison does not search over lattice
    correspondences - only the input cells of the atoms objects are considered.

    atoms1:               The first atoms object.
    atoms2:               The second atoms object.
    frame:                How to construct the common frame of reference.
                          'left': use the cell of atoms1.
                          'right' use the cell of atoms1.
                          'central': use the cell which is halfway between
                                     left and right cells.
                          Default is 'central'.
    ignore_stoichiometry: Whether to use stoichiometry in comparison.
                          Stoichiometries must be identical if 'True'.
    allow_reflection:     Whether to test mirror images.
    allow_rotation:       Whether to test rotations.  Only used for 1D
                          structures (for 2D and 3D structures the rotation
                          is implicitly contained in the intermediate cell).

    Returns
    =======

    result: namedtuple of type 'RMSDResult', containing the elements:
            rmsd:        the root-mean-square distance between atoms in the
                         intermediate cell.
            dcell:       a measure of the strain necessary to deform the input
                         cells into the intermediate cell.
            cell:        the intermediate cell.
            basis:       change-of-basis matrix of the fractional coordinates
                         of atoms1.
            translation: the optimal translation of the fractional coordinates
                         of atoms1.
            permutation: the permutation of atoms1
    """

    assert frame in ['left', 'right', 'central']
    assert len(atoms1) > 0
    assert len(atoms2) > 0

    # verify that pbc's are equal (and correct)
    assert (atoms1.pbc == atoms2.pbc).all()
    assert (atoms1.pbc >= 0).all()
    assert (atoms1.pbc <= 1).all()

    # determine dimensionality type from periodic boundary conditions
    pbc = atoms1.pbc
    dim = np.sum(pbc)
    if dim == 0:
        raise ValueError("Comparison not meaningful for an aperiodic cell")

    assert dim >= 1 and dim <= 3

    num_chain_steps = None
    if dim == 1:
        n1 = len(atoms1)
        n2 = len(atoms2)
        num_chain_steps = min(n1, n2)

        if n1 != n2:
            lcm = np.lcm(n1, n2)
            atoms1 = atoms1 * [1, 1, lcm // n1]
            atoms2 = atoms2 * [1, 1, lcm // n2]
            #TODO: find a meaningful way of representing this in results.  perhaps use multiplication vector?

    res = _calculate_rmsd(atoms1, atoms2, frame, ignore_stoichiometry, 1,
                          allow_rotation, num_chain_steps)
    if not allow_reflection:
        return res

    atoms1 = atoms1.copy()
    scaled = -atoms1.get_scaled_positions(wrap=0)
    atoms1.set_scaled_positions(scaled)

    fres = _calculate_rmsd(atoms1, atoms2, frame, ignore_stoichiometry, -1,
                           allow_rotation, num_chain_steps)
    if fres.rmsd < res.rmsd:
        return fres
    else:
        return res
