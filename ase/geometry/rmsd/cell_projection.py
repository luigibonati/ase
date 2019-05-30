import numpy as np
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

    tol = 1E-10
    for cell in [a, b]:
        lengths = np.linalg.norm(cell, axis=1)
        if min(lengths) < tol:
            raise Exception("Unit cell vectors may not have zero length")

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


def parallelogram_area(a, b):
    return np.linalg.norm(np.cross(a, b))


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


def intermediate_representation(a, b, frame):

    imcell = calculate_intermediate_cell(a.cell, b.cell, frame)

    dim = sum(a.pbc)
    if dim == 1:
        # scale in axis direction
        for atoms in [a, b]:
            k = np.linalg.norm(imcell[2]) / np.linalg.norm(atoms.cell[2])
            positions = atoms.get_positions()
            atoms.set_positions(k * positions)
            atoms.set_cell(imcell, scale_atoms=False)

    elif dim == 2:
        # scale in out-of-plane direction
        imarea = parallelogram_area(imcell[0], imcell[1])

        for atoms in [a, b]:
            original = atoms.get_positions()

            area = parallelogram_area(atoms.cell[0], atoms.cell[1])
            target = np.sqrt(imarea / area)

            atoms.set_cell(imcell, scale_atoms=True)
            positions = atoms.get_positions()
            positions[:, 2] = original[:, 2] * target
            atoms.set_positions(positions)

    elif dim == 3:
        for atoms in [a, b]:
            atoms.set_cell(imcell, scale_atoms=True)

    # perform a minkowski reduction of the intermediate cell
    mr_cell, mr_path = minkowski_reduce(imcell, dim)
    a.set_cell(mr_cell, scale_atoms=False)
    b.set_cell(mr_cell, scale_atoms=False)

    celldist = cell_distance(a, b, frame)
    return a, b, celldist, mr_path
