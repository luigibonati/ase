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


def perpendicular_vector(x):

    tol = 1E-10
    norm = np.linalg.norm(x)
    if norm < tol:
        raise Exception("Input vector has zero length.")

    x = x / norm
    while 1:
        y = np.random.uniform(-1, 1, x.shape)
        norm = np.linalg.norm(y)
        if norm < 0.1:
            continue

        y /= norm
        y -= np.dot(x, y) * x
        norm = np.linalg.norm(y)
        if norm < 0.2:
            continue

        y /= norm
        if abs(np.dot(x, y)) < tol:
            return y


#TODO: pick a sensible name for this function
def correct_zero_length_vectors(atoms):

    dim = sum(atoms.pbc)
    if dim == 3:
        return

    tol = 1E-10
    cell = np.copy(atoms.cell)
    lengths = np.linalg.norm(atoms.cell, axis=1)
    indices = np.where(lengths < tol)[0]

    if dim == 2:
        sign = np.sign(np.linalg.det(cell))
        area = parallelogram_area(cell[0], cell[1])
        cell[2] = np.cross(cell[0], cell[1])
        cell[2] /= np.linalg.norm(cell[2])
        cell[2] *= np.sqrt(area)
        if sign < 0:
            cell[2] *= -1
        pos = atoms.get_positions(wrap=0)
        atoms.set_cell(cell, scale_atoms=False)
        atoms.set_positions(pos)
        return

    # TODO: maintain handedness
    if dim == 1:
        index = 2
        cell[0] = perpendicular_vector(cell[index])
        v = np.cross(cell[index], cell[0])
        cell[1] = v / np.linalg.norm(v)

        cell[0] *= np.linalg.norm(cell[2])
        cell[1] *= np.linalg.norm(cell[2])
        atoms.set_cell(cell, scale_atoms=False)
        return

    if len(indices) == 0:
        return

    if (dim == 1 and 2 in indices) or\
       (dim == 2 and min(indices) < 2) or\
       (dim == 3 and len(indices)):
        raise Exception("Cell vector in periodic direction has zero length.")

    if dim == 2:
        v = np.cross(cell[0], cell[1])
        cell[2] = v / np.linalg.norm(v)

    elif dim == 1:
        if len(indices) == 1:
            index = indices[0]
            i, j = list({0, 1, 2} - {index})
            v = np.cross(cell[i], cell[j])
            v /= np.linalg.norm(v)
            cell[index] = v
        else:
            i, j = indices
            index = list({0, 1, 2} - {i, j})[0]
            cell[i] = perpendicular_vector(cell[index])
            v = np.cross(cell[index], cell[i])
            cell[j] = v / np.linalg.norm(v)

    atoms.set_cell(cell, scale_atoms=False)


def calculate_intermediate_cell_chain(a, b, frame):

    if frame == 'left':
        return a
    elif frame == 'right':
        return b

    z = (np.linalg.norm(a[2]) + np.linalg.norm(b[2])) / 2
    #amax = max(np.linalg.norm(a[:2], axis=1))
    #bmax = max(np.linalg.norm(b[:2], axis=1))
    #l = max(amax, bmax)
    return np.diag([z, z, z])


def adjust_r_scale(pos, cell, imcell):

    k = np.linalg.norm(imcell[0]) / np.linalg.norm(cell[0])

    adjusted_pos = np.copy(pos)
    adjusted_pos[:2] *= k
    return adjusted_pos


def intermediate_representation(a, b, frame):

    yep = a.get_positions(wrap=False)
    for atoms in [a, b]:
        correct_zero_length_vectors(atoms)
        #print("#", np.linalg.norm(atoms.cell, axis=1))
        #print(atoms.cell.array)

    #print(a.get_scaled_positions())
    #print(b.get_scaled_positions())
    apos0 = a.get_positions(wrap=False)
    bpos0 = b.get_positions(wrap=False)
    #print(apos0 - bpos0)

    dim = sum(a.pbc)
    imcell = calculate_intermediate_cell(a.cell, b.cell, frame)
    #print(imcell)
    #print(np.linalg.norm(imcell[2]), np.linalg.norm(a.cell[2]), np.linalg.norm(b.cell[2]))

    if dim == 1:
        imcell = calculate_intermediate_cell_chain(a.cell, b.cell, frame)
        print("imcell:")
        print(imcell)
        print(np.linalg.norm(imcell, axis=1))

        yep0 = np.array([e / np.linalg.norm(e) for e in a.cell])
        yep1 = np.array([e / np.linalg.norm(e) for e in imcell])
        #print("!", np.dot(yep0, yep0.T))
        #print("!", np.dot(yep1, yep1.T))

        for atoms in [a, b]:
            k = np.linalg.norm(imcell[2]) / np.linalg.norm(atoms.cell[2])
            #scaled = atoms.get_scaled_positions() * k
            #atoms
            atoms.set_cell(imcell, scale_atoms=1)
            #atoms.set_positions(np.dot(atoms.get_positions() * k, yep1))

            #scaled = atoms.get_scaled_positions()
            #adjusted = adjust_r_scale(scaled, atoms.cell, imcell)
            #atoms.cell = imcell
            #atoms.set_scaled_positions(adjusted)
        print(a.get_positions() - b.get_positions())
        asdf
    else:
        for atoms in [a, b]:
            atoms.set_cell(imcell, scale_atoms=True)

    apos1 = a.get_positions(wrap=False)
    bpos1 = b.get_positions(wrap=False)
    print(a.get_positions() - b.get_positions())
    print("yes:", np.linalg.norm(a.get_positions() - b.get_positions()))

    affine1 = lstsq(apos0, apos1)[0]
    affine2 = lstsq(bpos0, bpos1)[0]

    # perform a minkowski reduction of the intermediate cell
    mr_cell, mr_path = minkowski_reduce(imcell, dim)
    a.set_cell(mr_cell, scale_atoms=False)
    b.set_cell(mr_cell, scale_atoms=False)

    celldist = cell_distance(a, b, frame)
    return a, b, celldist, mr_path, affine1, affine2
