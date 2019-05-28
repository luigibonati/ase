import numpy as np
from scipy.linalg import polar
from ase.geometry.rmsd.minkowski_reduction import gauss
from ase.geometry.rmsd.minkowski_reduction import reduce_basis


def apply_minkowski_reduction(atoms, mr_cell):

    apos = atoms.get_positions()
    atoms.cell = mr_cell
    atoms.set_positions(apos)
    atoms.wrap()


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


def parallelogram_area(a, b):
    return np.linalg.norm(np.cross(a, b))


def adjust_z_scale(pos, cell, imcell, imarea):

    area = parallelogram_area(cell[0], cell[1])
    k = (np.sqrt(imarea / area) * cell[2, 2] / imcell[2, 2])

    adjusted_pos = np.copy(pos)
    adjusted_pos[:, 2] *= k
    return adjusted_pos


def adjust_r_scale(pos, cell, imcell):

    k = np.linalg.norm(imcell[0]) / np.linalg.norm(cell[0])

    adjusted_pos = np.copy(pos)
    adjusted_pos[:2, :2] *= k
    return adjusted_pos


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


def calculate_intermediate_cell_monolayer(a, b, frame):

    if frame == 'left':
        return a
    elif frame == 'right':
        return b

    imcell = np.zeros((3, 3))
    imcell[:2, :2] = calculate_intermediate_cell(a[:2, :2], b[:2, :2], frame)
    imcell[2, 2] = max(a[2, 2], b[2, 2])
    return imcell


def calculate_intermediate_cell_chain(a, b, frame):

    if frame == 'left':
        return a
    elif frame == 'right':
        return b

    z = (a[2, 2] + b[2, 2]) / 2
    l = np.max(np.concatenate((a[:2, :2], b[:2, :2])).reshape(-1))
    imcell = np.diag([l, l, z])
    return imcell


def cell_distance(atoms0, atoms1, frame, scale_invariant=False):

    assert (atoms0.pbc == atoms1.pbc).all()
    dim = sum(atoms0.pbc)

    a = atoms0.cell
    b = atoms1.cell

    if dim == 1:
        a = a[2:, 2:]
        b = b[2:, 2:]
    elif dim == 2:
        a = a[:2, :2]
        b = b[:2, :2]

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


def intermediate_representation(a, b, frame):

    dim = sum(a.pbc)
    if dim == 1:
        imcell = calculate_intermediate_cell_chain(a.cell, b.cell, frame)

        for atoms in [a, b]:
            scaled = atoms.get_scaled_positions()
            adjusted = adjust_r_scale(scaled, atoms.cell, imcell)
            atoms.cell = imcell
            atoms.set_scaled_positions(adjusted)

    elif dim == 2:
        imcell = calculate_intermediate_cell_monolayer(a.cell, b.cell, frame)
        imarea = parallelogram_area(imcell[0], imcell[1])

        for atoms in [a, b]:
            scaled = atoms.get_scaled_positions()
            adjusted = adjust_z_scale(scaled, atoms.cell, imcell, imarea)
            atoms.cell = imcell
            atoms.set_scaled_positions(adjusted)

    elif dim == 3:
        imcell = calculate_intermediate_cell(a.cell, b.cell, frame)

        for atoms in [a, b]:
            scaled = atoms.get_scaled_positions()
            atoms.cell = imcell
            atoms.set_scaled_positions(scaled)

    # perform a minkowski reduction of the intermediate cell
    mr_path = np.eye(3).astype(np.int)
    if dim >= 2:
        mr_cell, mr_path = minkowski_reduce(imcell, dim)
        apply_minkowski_reduction(a, mr_cell)
        apply_minkowski_reduction(b, mr_cell)

    celldist = cell_distance(a, b, frame)
    return a, b, celldist, mr_path
