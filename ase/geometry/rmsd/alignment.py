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


def align(p0, eindices, p1_nbrs, nbr_cells):

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


# TODO: call ase.geometry.wrap_positions instead
def wrap(positions, cell, pbc):

    scaled = np.linalg.solve(cell.T, positions.T).T

    for i, periodic in enumerate(pbc):
        if periodic:
            # Yes, we need to do it twice.
            # See the scaled_positions.py test.
            scaled[:, i] %= 1.0
            scaled[:, i] %= 1.0

    return np.dot(scaled, cell)


def cherry_pick(pbc, imcell, s0, shift, nbr_cells, eindices, p1_nbrs,
                xindices, yindices, zindices, shift_counts):

    cindices = [xindices, yindices, zindices]
    num_atoms = len(s0)
    s0 = np.copy(s0)
    for i, index in enumerate(shift_counts):
        indices = cindices[i][:index]
        s0[indices] += shift[i]
        s0 -= index * shift[i] / num_atoms
        if pbc[i]:
            s0[:, i] % 1.0

    p0 = np.dot(s0, imcell)
    return align(p0, eindices, p1_nbrs, nbr_cells)


def slide(s0, scaled_shift, num_atoms, pbc, index, i):

    s0[index] += scaled_shift[i]
    s0 -= scaled_shift[i] / num_atoms
    if pbc[i]:
        s0[:, i] % 1.0


def find_chain_alignment(dim, pbc, imcell, s0, p1, scaled_shift, eindices,
                         zindices, cell_length, num_chain_steps):

    num_atoms = len(s0)
    assert dim == 1
    if num_chain_steps is not None:
        zindices = zindices[:num_chain_steps]

    best = (float('inf'), None, None, None)

    for kk, k in enumerate(zindices):

        p0 = np.dot(s0, imcell)
        b = (best[0], np.arange(num_atoms), np.eye(2))
        rmsd, perm, u = register_chain(p0, p1, eindices, cell_length, b)
        U = np.eye(3)
        U[:2, :2] = u

        trial = (rmsd, perm, U, (0, 0, kk))
        best = min(best, trial, key=lambda x: x[0])
        slide(s0, scaled_shift, num_atoms, pbc, k, 2)

    rmsd, perm, U, ijk = best
    return rmsd, perm, U, ijk


def find_alignment(dim, pbc, imcell, s0, p1, scaled_shift, nbr_cells, eindices,
                   p1_nbrs, xindices, yindices, zindices):

    num_atoms = len(s0)
    best = (float('inf'), None, None)

    for kk, k in enumerate(zindices):
        for jj, j in enumerate(yindices):
            for ii, i in enumerate(xindices):

                p0 = np.dot(s0, imcell)
                rmsd, perm = align(p0, eindices, p1_nbrs, nbr_cells)
                trial = (rmsd, perm, (ii, jj, kk))
                best = min(best, trial, key=lambda x: x[0])

                slide(s0, scaled_shift, num_atoms, pbc, i, 0)
            slide(s0, scaled_shift, num_atoms, pbc, j, 1)
        slide(s0, scaled_shift, num_atoms, pbc, k, 2)

    rmsd, perm, ijk = best
    return rmsd, perm, np.eye(3), ijk


class LatticeComparator:

    def __init__(self, atoms0, atoms1):

        self.pbc = atoms0.pbc
        self.dim = sum(self.pbc)
        self.imcell = atoms0.cell

        # get centred atomic positions
        p0 = atoms0.get_positions()
        p1 = atoms1.get_positions()
        self.mean0 = np.mean(p0, axis=0)
        self.mean1 = np.mean(p1, axis=0)
        p0 = wrap(p0 - self.mean0, self.imcell, self.pbc)
        p1 = wrap(p1 - self.mean1, self.imcell, self.pbc)
        self.p0 = p0
        self.p1 = p1

        self.s0 = np.linalg.solve(self.imcell.T, p0.T).T

        self.numbers = atoms0.numbers
        self.eindices = [np.where(self.numbers == element)[0]
                         for element in np.unique(self.numbers)]

        self.shift = get_shift_vectors(self.dim, self.imcell)
        self.scaled_shift = np.linalg.solve(self.imcell.T, self.shift.T).T

        self.nbr_cells = get_neighboring_cells(self.dim, self.imcell)
        self.p1_nbrs = [np.concatenate([p1[indices] + np.dot(self.shift.T, nbr)
                        for nbr in self.nbr_cells])
                        for indices in self.eindices]

        xindices, yindices, zindices = np.argsort(self.s0, axis=0).T
        if self.dim == 1:
            xindices = [0]
            yindices = [0]
        elif self.dim == 2:
            zindices = [0]

        self.xindices = xindices
        self.yindices = yindices
        self.zindices = zindices

    def best_alignment(self, allow_rotation, num_chain_steps):

        if allow_rotation:
            assert self.dim == 1
            cell_length = self.imcell[2, 2]
            res = find_chain_alignment(self.dim, self.pbc, self.imcell,
                                       self.s0, self.p1, self.scaled_shift,
                                       self.eindices, self.zindices,
                                       cell_length, num_chain_steps)
        else:
            res = find_alignment(self.dim, self.pbc, self.imcell, self.s0,
                                 self.p1, self.scaled_shift, self.nbr_cells,
                                 self.eindices, self.p1_nbrs,
                                 self.xindices, self.yindices, self.zindices)
        rmsd, perm, U, ijk = res

        num_atoms = len(self.numbers)
        i, j, k = ijk
        translation = -np.dot(U, self.mean0) + self.mean1
        translation -= i * self.shift[0] / num_atoms
        translation -= j * self.shift[1] / num_atoms
        translation -= k * self.shift[2] / num_atoms

        return rmsd, perm, U, translation

    def cherry_pick(self, c):
        return cherry_pick(self.pbc, self.imcell, self.s0, self.scaled_shift,
                           self.nbr_cells, self.eindices, self.p1_nbrs,
                           self.xindices, self.xindices, self.zindices, c)
