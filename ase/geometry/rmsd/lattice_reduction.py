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

	for c in zip(*indices):
		p0 = lr.get_point(c)
		for i, e in enumerate(p0):
			uf.merge(i, e)

	return uf.get_components(relabel=True)


# TODO: remove this if/when merged into Atoms object
def _pretty_translation(scaled, pbc, eps):

	scaled = np.copy(scaled)
	for i in range(3):
		if not pbc[i]:
			continue

		indices = np.argsort(scaled[:, i])
		sp = scaled[indices, i]

		widths = (np.roll(sp, 1) - sp)
		indices = np.where(widths < -eps)[0]
		widths[indices] %= 1.0
		scaled[:, i] -= sp[np.argmin(widths)]

	indices = np.where(scaled < -eps)
	scaled[indices] %= 1.0
	return scaled


def pretty_translation(atoms, eps=1E-7):
	"""Translates atoms such that scaled positions are minimized."""

	scaled = atoms.get_scaled_positions()
	pbc = atoms.pbc

	# Don't use the tolerance unless it gives better results
	s0 = _pretty_translation(scaled, pbc, 0)
	s1 = _pretty_translation(scaled, pbc, eps)
	if np.max(s0) < np.max(s1) + eps:
		scaled = s0
	else:
		scaled = s1

	atoms.set_scaled_positions(scaled)


def reduced_layout(n, dim, H, lr, rmsd):

	#TODO: could also verify that elements are identical

	if dim == 2:
		_H = np.diag([n, n, n])
		_H[:dim, :dim] = H
	elif dim == 3:
		_H = np.copy(H)
	else:
		raise Exception("not implemented yet")
	rcell = np.dot(_H, lr.imcell) / n

	mr_cell, _ = minkowski_reduce(rcell, dim)
	absolute = np.dot(lr.s0, lr.imcell)
	clustered = Atoms(positions=absolute, cell=mr_cell, numbers=lr.numbers, pbc=lr.pbc)
	clustered.wrap()
	#view(clustered, block=1)
	#asdf


	indices = get_group_elements(n, dim, H)
	components = group_atoms(n, lr, indices)
	nbr_cells = alignment.get_neighboring_cells(dim, clustered.cell)

	shift = alignment.get_shift_vectors(dim, clustered.cell)
	shifts = [np.dot(shift.T, nbr) for nbr in nbr_cells]

	rmsd_check = 0
	csizes = []
	clustered_numbers = []
	clustered_positions = []
	for c in np.unique(components):

		atom_indices = np.where(c == components)[0]
		csizes.append(len(atom_indices))
		i = atom_indices[0]

		positions = []
		ps = clustered.get_positions()
		for c in zip(*indices):
			j = lr.get_point(c)[i]
			ds = np.linalg.norm(ps[i] - (ps[j] + shifts), axis=1)
			shift_index = np.argmin(ds)
			positions.append(ps[j] + shifts[shift_index])

		#average atoms
		clustered_positions.append(np.mean(positions, axis=0))
		clustered_numbers.append(clustered.numbers[i])
		rmsd_check += np.sum(cdist(positions, positions, 'sqeuclidean'))
	rmsd_check = np.sqrt(rmsd_check / n)

	if len(np.unique(csizes)) > 1:
		return None

	tol = 1E-12
	if abs(rmsd - rmsd_check) > tol:
		return None

	print("check:", H.reshape(-1), rmsd, rmsd_check, abs(rmsd - rmsd_check) < 1E-12, len(np.unique(csizes)), np.prod([n // e for e in np.diag(H)]))

	reduced = Atoms(positions=clustered_positions, numbers=clustered_numbers, cell=rcell, pbc=lr.pbc)
	reduced.wrap()
	pretty_translation(reduced)
	return reduced


def find_lattice_reductions(atoms, keep_all=False):

	Reduced = namedtuple('ReducedLattice', 'rmsd factor atoms')

	n = len(atoms)
	dim = sum(atoms.pbc)
	reductions, lr = find_consistent_reductions(atoms)

	reduced = {}
	for i, (rmsd, group_index, H) in enumerate(reductions):

		print()
		reduced_atoms = reduced_layout(n, dim, H, lr, rmsd)
		if reduced_atoms is None:
			continue

		if rmsd > 0.001: continue

		group_index = np.prod(n // np.diag(H))
		key = [group_index, i][keep_all]
		entry = Reduced(rmsd=rmsd, factor=group_index, atoms=reduced_atoms)
		if key not in reduced:
			reduced[key] = entry
		else:
			reduced[key] = min(reduced[key], entry, key=lambda x: x.rmsd)

	return sorted(reduced.values(), key=lambda x: x.factor, reverse=1)#[:1]
