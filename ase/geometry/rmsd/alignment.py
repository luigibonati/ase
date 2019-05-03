import numpy as np
from scipy.spatial.distance import cdist
from ase.geometry.rmsd.chain_alignment import register_chain
from ase.geometry.rmsd.assignment import linear_sum_assignment


def concatenate_permutations(perms):

    perm = []
    for p in perms:
        perm.extend(list(p + len(perm)))
    return np.array(perm)


def get_shift_vectors(dim, imcell):

    shift = np.copy(imcell)
    if dim == 1:
        shift[0] *= 0
        shift[1] *= 0
    elif dim == 2:
        shift[2] *= 0

    return shift


def get_neighboring_cells(dim, imcell):

    # Minkowski reduction means closest neighbours are sufficient
    xlim = ylim = zlim = 1
    if dim == 1:
        xlim = 0
        ylim = 0
    elif dim == 2:
        zlim = 0

    nbr_cells = [(i, j, k) for i in range(-xlim, xlim + 1)
                 for j in range(-ylim, ylim + 1)
                 for k in range(-zlim, zlim + 1)]
    return nbr_cells


def align(p0, p1, eindices, p1_nbrs, nbr_cells):

    nrmsdsq = 0
    perms = []
    for indices, pmatch in zip(eindices, p1_nbrs):

        num = len(indices)
        dist = cdist(p0[indices], pmatch, 'sqeuclidean')
        dist = np.min(dist.reshape((num, len(nbr_cells), num)), axis=1)
        res = linear_sum_assignment(dist)
        perms.append(res[1])
        nrmsdsq += np.sum(dist[res])

    num_atoms = len(p0)
    rmsd = np.sqrt(nrmsdsq / num_atoms)
    return rmsd, concatenate_permutations(perms)


def wrap(positions, cell, pbc):

    scaled = np.linalg.solve(cell.T, positions.T).T

    for i, periodic in enumerate(pbc):
        if periodic:
            # Yes, we need to do it twice.
            # See the scaled_positions.py test.
            scaled[:, i] %= 1.0
            scaled[:, i] %= 1.0

    return np.dot(scaled, cell)


def slide(s0, shift, num_atoms, pbc, index, i):

    s0[index] += shift[i]
    s0 -= shift[i] / num_atoms
    if pbc[i]:
        s0[:, i] % 1.0


def find_alignment(dim, pbc, imcell, s0, p1, shift, nbr_cells, eindices,
                   p1_nbrs, numbers, cell_length, allow_rotation,
                   num_chain_steps):

    num_atoms = len(s0)

    xindices = np.argsort(s0[:, 0])
    yindices = np.argsort(s0[:, 1])
    zindices = np.argsort(s0[:, 2])

    if dim == 1:
        xindices = [0]
        yindices = [0]
        if num_chain_steps is not None:
            zindices = zindices[:num_chain_steps]
    elif dim == 2:
        zindices = [0]

    U = np.eye(3)
    best = (float('inf'), None, None, None)

    for kk, k in enumerate(zindices):
        for jj, j in enumerate(yindices):
            for ii, i in enumerate(xindices):

                p0 = np.dot(s0, imcell)

                if dim == 1 and allow_rotation:

                    b = (best[0], np.arange(num_atoms), np.eye(2))
                    rmsd, perm, u = register_chain(p0, p1, eindices,
                                                   cell_length, b)
                    U = np.eye(3)
                    U[:2, :2] = u
                else:
                    rmsd, perm = align(p0, p1, eindices, p1_nbrs, nbr_cells)

                trial = (rmsd, perm, U, (ii, jj, kk))
                best = min(best, trial, key=lambda x: x[0])

                slide(s0, shift, num_atoms, pbc, i, 0)
            slide(s0, shift, num_atoms, pbc, j, 1)
        slide(s0, shift, num_atoms, pbc, k, 2)

    rmsd, perm, U, ijk = best
    return rmsd, perm, U, ijk


def best_alignment(atoms0, atoms1, allow_rotation, num_chain_steps):

    pbc = atoms0.pbc
    dim = sum(pbc)
    imcell = atoms0.cell

    # get centred atomic positions
    p0 = atoms0.get_positions()
    p1 = atoms1.get_positions()
    mean0 = np.mean(p0, axis=0)
    mean1 = np.mean(p1, axis=0)
    p0 = wrap(p0 - mean0, imcell, pbc)
    p1 = wrap(p1 - mean1, imcell, pbc)

    s0 = np.linalg.solve(imcell.T, p0.T).T

    numbers = atoms0.numbers
    elements = np.unique(numbers)
    eindices = [np.where(numbers == element)[0] for element in elements]

    shift = get_shift_vectors(dim, imcell)
    scaled_shift = np.linalg.solve(imcell.T, shift.T).T

    nbr_cells = get_neighboring_cells(dim, imcell)
    p1_nbrs = [np.concatenate([p1[indices] + np.dot(shift.T, nbr)
               for nbr in nbr_cells]) for indices in eindices]

    cell_length = imcell[2, 2]
    rmsd, perm, U, ijk = find_alignment(dim, pbc, imcell, s0, p1, scaled_shift,
                                        nbr_cells, eindices, p1_nbrs, numbers,
                                        cell_length, allow_rotation,
                                        num_chain_steps)

    num_atoms = len(numbers)
    i, j, k = ijk
    translation = -np.dot(U, mean0) + mean1
    translation -= i * shift[0] / num_atoms
    translation -= j * shift[1] / num_atoms
    translation -= k * shift[2] / num_atoms

    return rmsd, perm, U, translation
