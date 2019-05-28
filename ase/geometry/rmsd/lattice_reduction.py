import numpy as np
from scipy.spatial.distance import cdist
from collections import namedtuple

from ase import Atoms
import ase.geometry.rmsd.alignment as alignment
from ase.geometry.rmsd.cell_projection import minkowski_reduce
from ase.geometry.dimensionality.disjoint_set import DisjointSet
from ase.visualize import view
from ase.geometry.rmsd.lattice_subgroups import find_consistent_reductions, get_group_elements


def group_atoms(n, lr, indices):

	uf = DisjointSet(n)

	for c in indices:
		p0 = lr.get_point(c)
		for i, e in enumerate(p0):
			uf.merge(i, e)

	components = uf.get_components()
	assert (lr.numbers == lr.numbers[components]).all()
	return uf.get_components(relabel=True)


def clustered_atoms(n, dim, H, lr):

	if dim == 1:
		raise Exception("not implemented yet")
	elif dim == 2:
		_H = np.diag([n, n, n])
		_H[:dim, :dim] = H
	elif dim == 3:
		_H = np.copy(H)

	rcell = np.dot(_H, lr.imcell) / n
	mr_cell, _ = minkowski_reduce(rcell, dim)

	positions = np.dot(lr.s0, lr.imcell)
	clustered = Atoms(positions=positions, cell=mr_cell,
				numbers=lr.numbers, pbc=lr.pbc)
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
	components = group_atoms(n, lr, indices)
	if len(np.unique(np.bincount(components))) != 1:
		# TODO: verify what is going on here
		return None

	clustered = clustered_atoms(n, dim, H, lr)
	nbr_cells = alignment.get_neighboring_cells(dim, clustered.cell)
	shift = alignment.get_shift_vectors(dim, clustered.cell)
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

	print("check:", H.reshape(-1), rmsd, rmsd_check, np.prod([n // e for e in np.diag(H)]))

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

		group_index = np.prod(n // np.diag(H))
		key = [group_index, i][keep_all]
		entry = Reduced(rmsd=rmsd, factor=group_index, atoms=reduced_atoms)
		if key not in reduced:
			reduced[key] = entry
		else:
			reduced[key] = min(reduced[key], entry, key=lambda x: x.rmsd)

	return sorted(reduced.values(), key=lambda x: x.factor)
