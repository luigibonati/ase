import numpy as np
from collections import namedtuple
from ase.geometry.rmsd.cell_projection import intermediate_representation
from ase.geometry.rmsd.alignment import LatticeComparator
from ase.geometry.rmsd.standard_form import standardize_atoms


def invert_permutation(perm):
    return np.argsort(perm)


def _calculate_rmsd(atoms1, atoms2, frame, ignore_stoichiometry, sign,
                    allow_rotation, multiplier1, multiplier2,
                    num_chain_steps=None):

    a = atoms1.copy()
    b = atoms2.copy()

    res = standardize_atoms(a, b, ignore_stoichiometry)
    atomic_perms, axis_perm = res

    res = intermediate_representation(a, b, frame, allow_rotation)
    pa, pb, celldist, mr_path, linear_map1, linear_map2 = res

    lc = LatticeComparator(pa, pb)
    res = lc.best_alignment(allow_rotation, num_chain_steps)
    rmsd, assignment, U, translation = res

    # undo Minkowski-reduction
    inv_mr = np.linalg.inv(mr_path)
    imcell = np.dot(inv_mr, pa.cell)

    # undo atomic permutation
    left, right = atomic_perms
    invleft = invert_permutation(left)
    assignment = invert_permutation(right[assignment[invleft]])

    # undo axis permutation
    invaxis = invert_permutation(axis_perm)
    imcell = imcell[invaxis][:, invaxis]
    U = U[invaxis][:, invaxis]
    translation = translation[invaxis]
    linear_map1 = linear_map1[invaxis][:, invaxis].T
    linear_map2 = linear_map2[invaxis][:, invaxis].T

    # convert basis and translation to fractional coordinates
    fractional = np.linalg.solve(imcell.T, translation.T).T
    basis = np.dot(imcell, np.linalg.solve(imcell.T, U.T).T)
    basis *= sign

    affine1 = np.zeros((4, 4))
    affine2 = np.zeros((4, 4))

    affine1[:3, :3] = np.dot(linear_map1, U) * sign
    affine1[:3, 3] = translation
    affine2[:3, :3] = linear_map2

    entries = 'rmsd dcell cell basis translation permutation mul1 mul2 affine1 affine2'
    result = namedtuple('RMSDResult', entries)
    return result(rmsd=rmsd, dcell=celldist, cell=imcell, basis=basis,
                  translation=fractional, permutation=assignment,
                  mul1=multiplier1, mul2=multiplier2, affine1=affine1, affine2=affine2)


def calculate_rmsd(atoms1, atoms2, frame='central', ignore_stoichiometry=False,
                   allow_reflection=False, allow_rotation=False):
    """Calculates the optimal RMSD between two crystalline structures.

    A common frame of reference is calculated and the optimal translation is
    found.  The periodic boundary conditions of the input atoms objects
    determine the type of comparison to be made (along a line, in-plane, or a
    full three-dimensional).  Non-crystalline (or molecular) comparisons, i.e.
    comparisons with no periodicity, are not supported.  For 1D (chain-like)
    structures, the optimal rotation about the chain axis can optionally be
    found.  Note that for 2D and 3D structures the comparison does not search
    over lattice correspondences - only the input cells of the atoms objects
    are considered.  For 1D chains, the optimal lattice correspondence is
    determined by extending the cells along the chain axis, using the least
    common multiple of the Atoms objects.

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
            mul1:        the multiplication vector of atoms1 (only used for
                         chains of different sizes)
            mul2:        the multiplication vector of atoms2 (only used for
                         chains of different sizes)
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

    n1 = len(atoms1)
    n2 = len(atoms2)
    num_chain_steps = min(n1, n2)
    multiplier1 = [1, 1, 1]
    multiplier2 = [1, 1, 1]

    if dim == 1 and n1 != n2:
        lcm = np.lcm(n1, n2)
        multiplier1 = [1, 1, lcm // n1]
        multiplier2 = [1, 1, lcm // n2]
        atoms1 = atoms1 * multiplier1
        atoms2 = atoms2 * multiplier2

    res = _calculate_rmsd(atoms1, atoms2, frame, ignore_stoichiometry, 1,
                          allow_rotation, multiplier1, multiplier2,
                          num_chain_steps)
    if not allow_reflection:
        return res

    atoms1 = atoms1.copy()
    scaled = -atoms1.get_scaled_positions(wrap=0)
    atoms1.set_scaled_positions(scaled)

    fres = _calculate_rmsd(atoms1, atoms2, frame, ignore_stoichiometry, -1,
                           allow_rotation, multiplier1, multiplier2,
                           num_chain_steps)
    if fres.rmsd < res.rmsd:
        return fres
    else:
        return res
