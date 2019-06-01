import numpy as np
from scipy.linalg import polar, lstsq
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


def calculate_intermediate_cell_chain(a, b, frame):

    if frame == 'left':
        return a
    elif frame == 'right':
        return b

    z = (np.linalg.norm(a[2]) + np.linalg.norm(b[2])) / 2
    return z / np.linalg.norm(a[2]) * a


def intermediate_representation(a, b, frame, allow_rotation):

    apos0 = a.get_positions(wrap=False)
    bpos0 = b.get_positions(wrap=False)

    tol = 1E-10
    dim = sum(a.pbc)
    if dim == 1:
        av = a.cell[2] / np.linalg.norm(a.cell[2])
        bv = b.cell[2] / np.linalg.norm(b.cell[2])
        dot = np.dot(av, bv)
        axis_aligned = abs(dot - 1) < tol
        if not axis_aligned and not allow_rotation:
            raise Exception("Comparison not meaningful for cells with \
unaligned chain axes and allow_rotation=False")

        for atoms in [a, b]:
            cell = np.copy(atoms.cell)
            for i in range(2):
                cell[i] *= np.linalg.norm(cell[2]) / np.linalg.norm(cell[i])
            atoms.set_cell(cell, scale_atoms=False)

    elif dim == 2:
        for atoms in [a, b]:
            for i in range(2):
                cell = np.copy(atoms.cell)
                cell[2] = np.cross(cell[0], cell[1])
                cell[2] /= np.sqrt(np.linalg.norm(cell[2]))
            atoms.set_cell(cell, scale_atoms=False)

    imcell = calculate_intermediate_cell(a.cell, b.cell, frame)

    if dim == 1 and not allow_rotation:
        imcell = calculate_intermediate_cell_chain(a.cell, b.cell, frame)

        for atoms in [a, b]:
            k = np.linalg.norm(imcell[2]) / np.linalg.norm(atoms.cell[2])
            positions = atoms.get_positions(wrap=False)
            atoms.set_cell(imcell, scale_atoms=False)
            atoms.set_positions(k * positions)
    else:
        for atoms in [a, b]:
            atoms.set_cell(imcell, scale_atoms=True)

    apos1 = a.get_positions(wrap=False)
    bpos1 = b.get_positions(wrap=False)

    affine1 = lstsq(apos0, apos1)[0]
    affine2 = lstsq(bpos0, bpos1)[0]

    # perform a minkowski reduction of the intermediate cell
    mr_cell, mr_path = minkowski_reduce(imcell, dim)
    a.set_cell(mr_cell, scale_atoms=False)
    b.set_cell(mr_cell, scale_atoms=False)

    celldist = cell_distance(a, b, frame)
    return a, b, celldist, mr_path, affine1, affine2
