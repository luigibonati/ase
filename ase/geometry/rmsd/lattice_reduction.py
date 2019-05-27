import numpy as np
from collections import namedtuple
from ase.geometry.rmsd.cell_projection import intermediate_representation
from ase.geometry.rmsd.alignment import best_alignment
from ase.geometry.rmsd.standard_form import standardize_atoms

import ase.geometry.rmsd.alignment as alignment


def invert_permutation(perm):

	n = len(perm)
	invperm = np.arange(n)
	for i, e in enumerate(perm):
		invperm[e] = i
	return invperm


def slide(s0, shift, num_atoms, pbc, index, i):

	s0[index] += shift[i]
	s0 -= shift[i] / num_atoms
	if pbc[i]:
		s0[:, i] % 1.0


def cherry_pick(dim, pbc, imcell, s0, shift, nbr_cells, eindices, p1_nbrs, cindices, shift_counts):

	num_atoms = len(s0)
	s0 = np.copy(s0)
	q0 = np.copy(s0)
	for i, index in enumerate(shift_counts):
		indices = cindices[i][:index]
		q0[indices] += shift[i]
		q0 -= index * shift[i] / num_atoms
		if pbc[i]:
			q0[:, i] % 1.0

	p0 = np.dot(q0, imcell)
	return alignment.align(p0, eindices, p1_nbrs, nbr_cells)


class LatticeReducer:

	def __init__(self, atoms):

		# TODO: can some of this be streamlined?
		res = standardize_atoms(atoms.copy(), atoms.copy(), False)
		a, b, atomic_perms, axis_perm = res

		pa, pb, celldist, mr_path = intermediate_representation(a, b, 'central')
		# ###

		data = alignment.initialize(pa, pb)
		pbc, dim, imcell, mean0, mean1, s0, numbers, eindices, shift, scaled_shift, nbr_cells, p1_nbrs = data

		num_atoms = len(s0)
		xindices = np.argsort(s0[:, 0])
		yindices = np.argsort(s0[:, 1])
		zindices = np.argsort(s0[:, 2])

		if dim == 1:
			xindices = [0]
			yindices = [0]
		elif dim == 2:
			zindices = [0]

		nx, ny, nz = len(xindices), len(yindices), len(zindices)
		distances = np.zeros((nx, ny, nz))
		permutations = -np.ones((nx, ny, nz, num_atoms)).astype(np.int)

		self.cindices = [xindices, yindices, zindices]
		self.dim = dim
		self.pbc = pbc
		self.imcell = imcell
		self.s0 = s0
		self.scaled_shift = scaled_shift
		self.nbr_cells = nbr_cells
		self.eindices = eindices
		self.p1_nbrs = p1_nbrs

		for kk, k in enumerate(zindices):
			break
			for jj, j in enumerate(yindices):
				for ii, i in enumerate(xindices):

					p0 = np.dot(s0, imcell)
					rmsd, perm = alignment.align(p0, eindices, p1_nbrs, nbr_cells)
					distances[ii, jj, kk] = rmsd
					permutations[ii, jj, kk, :] = perm
					slide(s0, scaled_shift, num_atoms, pbc, i, 0)
				slide(s0, scaled_shift, num_atoms, pbc, j, 1)
			slide(s0, scaled_shift, num_atoms, pbc, k, 2)

		self.distances = distances
		self.permutations = permutations


	def get_point(self, ijk):

		i, j, k = ijk
		if self.permutations[i, j, k, 0] != -1:
			return self.permutations[i, j, k]

		rmsd, permutation = cherry_pick(self.dim, self.pbc, self.imcell, self.s0,
						self.scaled_shift, self.nbr_cells,
						self.eindices, self.p1_nbrs, self.cindices, ijk)
		self.distances[i, j, k] = rmsd
		self.permutations[i, j, k] = permutation
		return permutation


def find_lattice_reductions(atoms):

	res = standardize_atoms(atoms.copy(), atoms.copy(), False)
	a, b, atomic_perms, axis_perm = res

	pa, pb, celldist, mr_path = intermediate_representation(a, b, 'central')
	res = best_alignment(pa, pb, False, None, True)
	rmsd, assignment, U, translation, distances, permutations = res
	return distances, permutations

	'''
	# undo Minkowski-reduction
	inv_mr = np.linalg.inv(mr_path)
	imcell = np.dot(inv_mr, pa.cell)

	# convert basis and translation to fractional coordinates
	fractional = np.linalg.solve(imcell.T, translation.T).T
	basis = np.dot(imcell, np.linalg.solve(imcell.T, U.T).T)
	basis *= sign

	# undo atomic permutation
	left, right = atomic_perms
	invleft = invert_permutation(left)
	assignment = invert_permutation(right[assignment[invleft]])

	# undo axis permutation
	invaxis = invert_permutation(axis_perm)
	imcell = imcell[invaxis][:, invaxis]
	fractional = fractional[invaxis]
	basis = basis[invaxis][:, invaxis]

	entries = 'rmsd dcell cell basis translation permutation mul1 mul2'
	result = namedtuple('RMSDResult', entries)
	return result(rmsd=rmsd, dcell=celldist, cell=imcell, basis=basis,
				  translation=fractional, permutation=assignment,
				  mul1=multiplier1, mul2=multiplier2)
	'''
