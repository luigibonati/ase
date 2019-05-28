import itertools
import numpy as np
from scipy.spatial.distance import cdist
from ase import Atoms
from ase.geometry.rmsd.cell_projection import intermediate_representation
from ase.geometry.rmsd.standard_form import standardize_atoms
import ase.geometry.rmsd.alignment as alignment
from ase.geometry.dimensionality.disjoint_set import DisjointSet


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

	def __init__(self, atoms, lazy=True):

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

		if not lazy:
			for kk, k in enumerate(zindices):
				for jj, j in enumerate(yindices):
					for ii, i in enumerate(xindices):

						p0 = np.dot(s0, imcell)
						rmsd, perm = alignment.align(p0, eindices, p1_nbrs, nbr_cells)
						distances[ii, jj, kk] = rmsd
						permutations[ii, jj, kk, :] = perm
						slide(s0, scaled_shift, num_atoms, pbc, i, 0)
					slide(s0, scaled_shift, num_atoms, pbc, j, 1)
				slide(s0, scaled_shift, num_atoms, pbc, k, 2)

		self.cindices = [xindices, yindices, zindices]
		self.dim = dim
		self.pbc = pbc
		self.imcell = imcell
		self.s0 = s0
		self.shift = shift
		self.scaled_shift = scaled_shift
		self.nbr_cells = nbr_cells
		self.eindices = eindices
		self.p1_nbrs = p1_nbrs
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



def invert_permutation(p):
	return np.argsort(p)


def get_divisors(n):
	return [i for i in range(1, n+1) if n % i == 0]


def _P(n):
	gcd = np.gcd
	return sum([gcd(k, n) for k in range(1, n+1)])


def number_of_subgroups(dim, n):

	assert dim in [1, 2, 3]

	gcd = np.gcd
	ds = get_divisors(n)

	if dim == 1:
		return len(ds)
	elif dim == 2:
		return sum([gcd(a, b) for a in ds for b in ds])
	else:
		total = 0
		for a, b, c in itertools.product(ds, repeat=3):
			A = gcd(a, n // b)
			B = gcd(b, n // c)
			C = gcd(a, n // c)
			ABC = A * B * C

			X = ABC // gcd(a * n // c, ABC)
			total += ABC // X**2 * _P(X)
		return total


#TODO: test this function against old x-step function
#@profile
def get_group_elements(n, dim, H):

	assert dim in [1, 2, 3]

	c = np.array([0] * dim)
	x = np.zeros([n] * dim).astype(np.int)

	if dim == 1:
		while x[tuple(c)] == 0:
			x[tuple(c)] = 1
			c += H[0]
			c %= n
	elif dim == 2:
		while x[tuple(c)] == 0:
			while x[tuple(c)] == 0:
				x[tuple(c)] = 1
				c += H[0]
				c %= n
			x[tuple(c)] = 1
			c += H[1]
			c %= n
	else:
		size = 1
		for e in np.diag(H):
			if e != 0:
				size *= n // e

		indices = np.zeros((size, 3)).astype(np.int)
		indices[:, 0] = H[0, 0] * np.arange(size)

		if H[1, 1] != 0:
			k = n // H[0, 0]
			indices[:, 0] += H[1, 0] * (np.arange(size) // k)
			indices[:, 1] += H[1, 1] * (np.arange(size) // k)

		if H[2, 2] != 0:
			k = n // H[1, 1] * n // H[0, 0]
			indices[:, 0] += H[2, 0] * (np.arange(size) // k)
			indices[:, 1] += H[2, 1] * (np.arange(size) // k)
			indices[:, 2] += H[2, 2] * (np.arange(size) // k)
		indices %= n

	return tuple(list(zip(*indices)))


#@profile
def is_consistent(dim, H, lr):

	n = len(lr.s0)
	seen = -np.ones((3, n)).astype(np.int)

	indices = get_group_elements(n, dim, H)

	for c in zip(*indices):
		p0 = lr.get_point(c)
		invp0 = invert_permutation(p0)

		for i, e in enumerate(H):
			c1 = (c + e) % n
			p1 = lr.get_point(c1)

			val = p1[invp0]
			if seen[i][0] == -1:
				seen[i] = val
			elif (seen[i] != val).any():
				return -float("inf")

	return np.sqrt(np.sum(lr.distances[indices]**2))


def check_1D_consistency(dim, n, lr, a):

	H = np.zeros((dim, dim)).astype(np.int)
	H[0, 0] = a

	return is_consistent(dim, H, lr) >= 0


def consistent_first_rows(dim, n, ds, lr):

	for a in ds:
		if check_1D_consistency(dim, n, lr, a):
			yield a


def check_2D_consistency(dim, n, lr, row1, row2):

	H = np.zeros((dim, dim)).astype(np.int)
	H[0] = row1
	H[1] = row2
	return is_consistent(dim, H, lr) >= 0


def get_basis(dim, n, lr):

	assert dim in [1, 2, 3]

	gcd = np.gcd
	ds = get_divisors(n)

	if dim == 1:
		for d in consistent_first_rows(dim, n, ds, lr):
			yield np.array([[d]])
	elif dim == 2:

		for a in consistent_first_rows(dim, n, ds, lr):
			for b in ds:
				for t in range(gcd(a, n // b)):
					s = t * a // gcd(a, n // b)
					yield np.array([[a, 0], [s, b]])
	elif dim == 3:

		for a in consistent_first_rows(dim, n, ds, lr):

			for b, c in itertools.product(ds, repeat=2):

				A = gcd(a, n // b)
				B = gcd(b, n // c)
				C = gcd(a, n // c)
				ABC = A * B * C

				X = ABC // gcd(a * n // c, ABC)

				for t in range(A):

					s = a * t // A

					consistent = check_2D_consistency(dim, n, lr, [a, 0, 0], [s, b, 0])
					if not consistent:
						continue

					for w in range(B * gcd(t, X) // X):

						v = b * X * w // (B * gcd(t, X))

						found = False
						for u in range(a):
							if (n // c * u) % a == (n * v * s // (b * c)) % a:
								u0 = u
								found = True
								break
						if not found:
							raise Exception("u not found")

						for z in range(C):
							u = u0 + a * z // C
							yield np.array([[a, 0, 0], [s, b, 0], [u, v, c]])


def find_consistent_reductions(atoms):

	n = len(atoms)
	dim = sum(atoms.pbc)
	r = number_of_subgroups(dim, n)
	print(n, r)

	lr = LatticeReducer(atoms)

	it = 0
	data = []
	for H in get_basis(dim, n, lr):

		#if np.sum(H) != np.trace(H): continue

		it += 1
		rmsd = is_consistent(dim, H, lr)
		if rmsd < 0:
			continue

		print(it, r, H.reshape(-1))

		group_index = n**dim // np.prod(np.diag(H))
		data.append((rmsd, group_index, H))

	print(n, len(data), r, it)
	return data, lr


def group_atoms(n, dim, H, lr):

	indices = get_group_elements(n, dim, H)

	uf = DisjointSet(n)

	for c in zip(*indices):
		p0 = lr.get_point(c)
		for i, e in enumerate(p0):
			uf.merge(i, e)

	return uf.get_components(relabel=True)


def pretty_translation(atoms):

	n = len(atoms)
	scaled = atoms.get_scaled_positions()

	for i in range(3):
		indices = np.argsort(scaled[:, i])
		sp = scaled[indices, i]
		widths = (np.roll(sp, 1) - sp) % 1.0
		scaled[:, i] -= sp[np.argmin(widths)]

	atoms.set_scaled_positions(scaled)
	atoms.wrap(eps=0)


def reduced_layout(n, dim, lr, components, rmsd, H, numbers):

	#TODO: could also verify that elements are identical

	indices = get_group_elements(n, dim, H)
	nbr_cells = alignment.get_neighboring_cells(dim, lr.imcell)
	shifts = [np.dot(lr.shift.T, nbr) for nbr in nbr_cells]

	rmsd_check = 0
	csizes = []

	clustered_numbers = []
	clustered_positions = []
	for c in np.unique(components):

		atom_indices = np.where(c == components)[0]
		csizes.append(len(atom_indices))

		ps = np.dot(lr.s0, lr.imcell)

		i = atom_indices[0]
		positions = []
		for c in zip(*indices):
			perm = lr.get_point(c)
			j = perm[i]
			pj = ps[j] - np.sum([lr.imcell[ii] * c[ii] / n for ii in range(3)], axis=0)

			ds = np.linalg.norm(ps[i] - (pj + shifts), axis=1)
			shift_index = np.argmin(ds)
			positions.append(pj + shifts[shift_index])

		#average atoms
		clustered_positions.append(np.mean(positions, axis=0))
		clustered_numbers.append(numbers[i])
		rmsd_check += np.sum(cdist(positions, positions, 'sqeuclidean'))

	rmsd_check = np.sqrt(rmsd_check / n)

	if len(np.unique(csizes)) > 1:
		return None

	tol = 1E-12
	if abs(rmsd - rmsd_check) > tol:
		return None

	rcell = np.dot(H, lr.imcell) / n
	print("check:", H.reshape(-1), rmsd, rmsd_check, abs(rmsd - rmsd_check) < 1E-12, len(np.unique(csizes)), np.prod([n // e for e in np.diag(H)]))

	reduced = Atoms(positions=clustered_positions, numbers=clustered_numbers, cell=rcell, pbc=lr.pbc)
	reduced.wrap()
	pretty_translation(reduced)

	from ase.visualize import view
	view(reduced, block=1)
	return reduced


def find_lattice_reductions(atoms, keep_all=False):

	n = len(atoms)
	dim = sum(atoms.pbc)
	reductions, lr = find_consistent_reductions(atoms)

	reduced = {}
	for i, (rmsd, group_index, H) in enumerate(reductions):

		components = group_atoms(n, dim, H, lr)

		reduced_atoms = reduced_layout(n, dim, lr, components, rmsd, H, atoms.numbers)
		if reduced_atoms is None:
			continue

		group_index = np.prod(n // np.diag(H))
		key = [group_index, i][keep_all]
		value = (rmsd, group_index, reduced_atoms)
		if key not in reduced:
			reduced[key] = value
		else:
			reduced[key] = min(reduced[key], value, key=lambda x:x[0])

	return sorted(reduced.values(), key=lambda x: x[1])

