import numpy as np
from collections import namedtuple
from scipy.spatial.distance import cdist

from ase import Atoms
from ase.geometry.rmsd.alignment import get_shift_vectors
from ase.geometry.rmsd.alignment import get_neighboring_cells
from ase.geometry.rmsd.lattice_subgroups import get_group_elements
from ase.geometry.rmsd.lattice_subgroups import find_consistent_reductions
from ase.geometry.rmsd.cell_projection import minkowski_reduce
from ase.geometry.dimensionality.disjoint_set import DisjointSet


def assign_atoms_to_clusters(n, lr, indices):

    uf = DisjointSet(n)

    for c in indices:
        p0 = lr.get_point(c)
        for i, e in enumerate(p0):
            uf.merge(i, e)

    components = uf.get_components()
    assert (lr.lc.numbers == lr.lc.numbers[components]).all()
    return uf.get_components(relabel=True)


def clustered_atoms(n, dim, H, lr):

    _H = np.diag([n, n, n])
    _H[:dim, :dim] = H

    rcell = np.dot(_H, lr.lc.imcell) / n
    mr_cell, _ = minkowski_reduce(rcell, dim)

    positions = np.dot(lr.lc.s0, lr.lc.imcell)
    clustered = Atoms(positions=positions, cell=mr_cell,
                      numbers=lr.lc.numbers, pbc=lr.lc.pbc)
    clustered.wrap()
    return clustered


def cluster_component(indices, ps, lr, shifts, i):

    positions = []
    for c in indices:
        j = lr.get_point(c)[i]
        ds = np.linalg.norm(ps[i] - (ps[j] + shifts), axis=1)
        shift_index = np.argmin(ds)
        positions.append(ps[j] + shifts[shift_index])

    meanpos = np.mean(positions, axis=0)
    component_rmsd = np.sum(cdist(positions, positions, 'sqeuclidean'))
    return meanpos, component_rmsd


def reduced_layout(n, dim, H, lr, rmsd):

    indices = get_group_elements(n, dim, H)
    components = assign_atoms_to_clusters(n, lr, indices)
    assert len(np.unique(np.bincount(components))) == 1

    clustered = clustered_atoms(n, dim, H, lr)
    nbr_cells = get_neighboring_cells(dim, clustered.cell)
    shift = get_shift_vectors(dim, clustered.cell)
    shifts = [np.dot(shift.T, nbr) for nbr in nbr_cells]

    data = []
    for c in np.unique(components):
        i = list(components).index(c)
        meanpos, crmsd = cluster_component(indices, clustered.get_positions(),
                                           lr, shifts, i)
        data.append((meanpos, clustered.numbers[i], crmsd))
    positions, numbers, crmsd = zip(*data)

    tol = 1E-12
    rmsd_check = np.sqrt(sum(crmsd) / n)
    if abs(rmsd - rmsd_check) > tol:
        return None

    reduced = Atoms(positions=positions, numbers=numbers,
                    cell=clustered.cell, pbc=clustered.pbc)
    reduced.wrap(pretty_translation=1)
    return reduced


def find_lattice_reductions(atoms, keep_all=False):

    Reduced = namedtuple('ReducedLattice', 'rmsd factor atoms')

    n = len(atoms)
    dim = sum(atoms.pbc)
    reductions, lr = find_consistent_reductions(atoms)

    reduced = {}
    for i, (rmsd, group_index, H) in enumerate(reductions):

        reduced_atoms = reduced_layout(n, dim, H, lr, rmsd)
        if reduced_atoms is None:
            continue

        rmsd /= 2 * n    # scaling from pairwise rmsd to cluster rmsd
        group_index = np.prod(n // np.diag(H))
        key = [group_index, i][keep_all]
        entry = Reduced(rmsd=rmsd, factor=group_index, atoms=reduced_atoms)
        if key not in reduced:
            reduced[key] = entry
        else:
            reduced[key] = min(reduced[key], entry, key=lambda x: x.rmsd)

        print(str(group_index).rjust(3), "%.4f" % rmsd, H.reshape(-1))

    return sorted(reduced.values(), key=lambda x: x.factor)
