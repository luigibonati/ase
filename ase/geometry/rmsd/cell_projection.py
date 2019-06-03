import numpy as np
import warnings
from scipy.linalg import polar
from ase.geometry.rmsd.minkowski_reduction import gauss
from ase.geometry.rmsd.minkowski_reduction import reduce_basis


def minkowski_reduce(cell, dim):

    mr_path = np.eye(3).astype(np.int)
    if dim == 1:
        mr_cell = np.copy(cell)

    elif dim == 2:
        hu, hv = gauss(cell, mr_path[0], mr_path[1])
        mr_path[0] = hu
        mr_path[1] = hv
        mr_cell = np.dot(mr_path, cell)

    elif dim == 3:
        _, mr_path = reduce_basis(cell)
        mr_cell = np.dot(mr_path, cell)
    return mr_cell, mr_path


def optimal_scale_factor(P):

    # s = (2*a00 + 2*a11 - a01**2 - a10**2) / (2*a00**2 + 2*a11**2)
    ts = np.trace(P**2)
    return (2 * np.trace(P) - np.sum(P**2) + ts) / (2 * ts)


def calculate_projection_matrix(a, b, scale_invariant=False):

    # calculate deformation gradient (linear map from A to B) and apply a polar
    # decomposition to remove the orthogonal (rotational) component of the
    # linear map
    F = np.linalg.solve(a, b)
    P = polar(F, side='right')[1]
    if scale_invariant:
        P *= optimal_scale_factor(P)

    # the intermediate cell is halfway mapping from A to B
    n = len(a)
    return (np.eye(n) + P) / 2


def calculate_intermediate_cell(a, b, frame):

    if frame == 'left':
        return a
    elif frame == 'right':
        return b

    M = calculate_projection_matrix(a, b)
    return np.dot(a, M)


def cell_distance(atoms0, atoms1, frame, scale_invariant=False):

    assert (atoms0.pbc == atoms1.pbc).all()
    dim = sum(atoms0.pbc)

    a = atoms0.cell
    b = atoms1.cell

    if dim == 1:
        a = polar(a[2:].T)[1]
        b = polar(b[2:].T)[1]
    elif dim == 2:
        a = polar(a[:2].T)[1]
        b = polar(b[:2].T)[1]

    mleft = calculate_projection_matrix(b, a, scale_invariant)
    mright = calculate_projection_matrix(a, b, scale_invariant)
    dleft = np.linalg.norm(mleft - np.eye(dim))
    dright = np.linalg.norm(mright - np.eye(dim))
    if frame == 'left':
        return dleft
    elif frame == 'right':
        return dright
    else:
        return (dleft + dright) / 2


def standardize_1D(a, b, allow_rotation):

    tol = 1E-10
    av = a.cell[2] / np.linalg.norm(a.cell[2])
    bv = b.cell[2] / np.linalg.norm(b.cell[2])
    dot = np.dot(av, bv)
    axis_aligned = abs(dot - 1) < tol

    if not axis_aligned:

        zero = False
        for atoms in [a, b]:
            lengths = np.linalg.norm(atoms.cell[:2], axis=1)
            zero |= (lengths < tol).any()

        if zero and not allow_rotation:
            raise Exception("Comparison meaningful for cells with unaligned \
chain axes and allow_rotation=False")

    Qs = []
    for atoms in [a, b]:
        cell = atoms.cell

        # check for orthogonality
        if np.dot(cell[0], cell[1]) >= tol:
            raise Exception("Off-axis cell vectors not orthogonal")

        if (np.dot(cell[: 2], cell[2]) >= tol).any():
            raise Exception("Off-axis cell vectors not perpendicular to axis \
cell vector")

        # fill in missing entries
        cell = cell.complete()

        # make off-axis cell vectors same length as axis vector
        for i in range(2):
            cell[i] *= np.linalg.norm(cell[2]) / np.linalg.norm(cell[i])
        atoms.set_cell(cell, scale_atoms=False)

        sign = np.sign(np.linalg.det(atoms.cell))
        Q, P = polar(sign * atoms.cell)
        P *= sign
        atoms.set_cell(P, scale_atoms=True)
        assert np.sign(np.linalg.det(atoms.cell)) == sign
        Qs.append(Q)
    return Qs


def standardize_2D(a, b):

    '''
        both out-of-plane cell vectors non-zero in both systems:
            warn if different handedness
            respect handedness

        one is non-zero:
            make other one have same handedness
            emit warning

        both zero:
            set both to right-handed
            don't warn
    '''

    tol = 1E-10
    zeros = [np.linalg.norm(atoms.cell[2]) < tol for atoms in [a, b]]

    if sum(zeros) == 1:

        tempa, tempb = a, b
        if not zeros[0]:
            tempa, tempb = tempb, tempa

        sign = np.sign(np.linalg.det(tempb.cell))
        assert sign != 0
        cell = tempa.cell
        cell[2] = np.cross(cell[0], cell[1])
        cell[2] *= sign
        warnings.warn("One cell has zero-length out-of-plane vector. \
The other does not.")

    for atoms in [a, b]:

        cell = atoms.cell
        if (np.abs(np.dot(cell[: 2], cell[2])) >= tol).any():
            raise Exception("Out-of-plane cell vector not perpendicular to \
in-plane vectors")

        sign = np.sign(np.linalg.det(cell))
        cell[2] = np.cross(cell[0], cell[1])
        cell[2] /= np.sqrt(np.linalg.norm(cell[2]))

        if sign != 0:
            cell[2] *= sign
        atoms.set_cell(cell, scale_atoms=False)


def standardize_cells(a, b, allow_rotation):

    tol = 1E-10
    for atoms in [a, b]:
        for i in range(3):
            if atoms.pbc[i] and np.linalg.norm(atoms.cell[i]) < tol:
                raise Exception("Cell vector in periodic direction has zero \
length")

    Qs = [np.eye(3)] * 2
    dim = sum(a.pbc)
    if dim == 1:
        Qs = standardize_1D(a, b, allow_rotation)
    elif dim == 2:
        standardize_2D(a, b)

    sa = np.sign(np.linalg.det(a.cell))
    sb = np.sign(np.linalg.det(b.cell))
    if sa != sb:
        warnings.warn('Cells have different handedness')
    return Qs


def intermediate_representation(a, b, frame, allow_rotation):

    Qs = standardize_cells(a, b, allow_rotation)

    imcell = calculate_intermediate_cell(a.cell, b.cell, frame)
    res1 = np.linalg.solve(a.cell, imcell)
    res2 = np.linalg.solve(b.cell, imcell)

    for atoms in [a, b]:
        atoms.set_cell(imcell, scale_atoms=True)

    linear_map1 = np.dot(Qs[0].T, res1)
    linear_map2 = np.dot(Qs[1].T, res2)

    # perform a minkowski reduction of the intermediate cell
    dim = sum(a.pbc)
    mr_cell, mr_path = minkowski_reduce(imcell, dim)
    a.set_cell(mr_cell, scale_atoms=False)
    b.set_cell(mr_cell, scale_atoms=False)

    celldist = cell_distance(a, b, frame)
    return a, b, celldist, mr_path, linear_map1, linear_map2
