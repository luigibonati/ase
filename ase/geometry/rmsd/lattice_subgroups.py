import itertools
import numpy as np

from ase.geometry.rmsd.cell_projection import intermediate_representation
from ase.geometry.rmsd.standard_form import standardize_atoms
import ase.geometry.rmsd.alignment as alignment


def cherry_pick(dim, pbc, imcell, s0, shift, nbr_cells, eindices, p1_nbrs, cindices, shift_counts):

	num_atoms = len(s0)
	s0 = np.copy(s0)
	for i, index in enumerate(shift_counts):
		indices = cindices[i][:index]
		s0[indices] += shift[i]
		s0 -= index * shift[i] / num_atoms
		if pbc[i]:
			s0[:, i] % 1.0

	p0 = np.dot(s0, imcell)
	return alignment.align(p0, eindices, p1_nbrs, nbr_cells)


class LatticeReducer:

	def __init__(self, atoms):

		res = standardize_atoms(atoms.copy(), atoms.copy(), False)
		a, b, atomic_perms, axis_perm = res
		pa, pb, celldist, mr_path = intermediate_representation(a, b, 'central')

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
		if dim == 3:
			distances = np.zeros((nx, ny, nz))
			permutations = -np.ones((nx, ny, nz, num_atoms)).astype(np.int)
		elif dim == 2:
			distances = np.zeros((nx, ny))
			permutations = -np.ones((nx, ny, num_atoms)).astype(np.int)
		else:
			raise Exception("not implemented yet")

		self.cindices = [xindices, yindices, zindices]
		self.dim = dim
		self.pbc = pbc
		self.imcell = imcell
		self.s0 = s0
		self.numbers = numbers
		self.shift = shift
		self.scaled_shift = scaled_shift
		self.nbr_cells = nbr_cells
		self.eindices = eindices
		self.p1_nbrs = p1_nbrs
		self.distances = distances
		self.permutations = permutations


	def get_point(self, ijk):

		ijk = tuple(ijk)
		if self.permutations[ijk][0] != -1:
			return self.permutations[ijk]

		rmsd, permutation = cherry_pick(self.dim, self.pbc, self.imcell, self.s0,
						self.scaled_shift, self.nbr_cells,
						self.eindices, self.p1_nbrs, self.cindices, ijk)
		self.distances[ijk] = rmsd
		self.permutations[ijk] = permutation
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


#TODO: clean this function up and test against old x-step function
def get_group_elements(n, dim, H):

	assert dim in [1, 2, 3]

	c = np.array([0] * dim)
	x = np.zeros([n] * dim).astype(np.int)

	if dim == 1:
		while x[tuple(c)] == 0:
			x[tuple(c)] = 1
			c += H[0]
			c %= n
		raise Exception("not implemented yet")

	elif dim == 2:
		while x[tuple(c)] == 0:
			while x[tuple(c)] == 0:
				x[tuple(c)] = 1
				c += H[0]
				c %= n
			x[tuple(c)] = 1
			c += H[1]
			c %= n
		indices = np.where(x)
		return list(zip(*indices))
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

	return indices


def is_consistent(dim, H, lr):

	n = len(lr.s0)
	seen = -np.ones((3, n)).astype(np.int)
	indices = get_group_elements(n, dim, H)

	for c in indices:
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

	indices = tuple(list(zip(*indices)))
	return np.sqrt(np.sum(lr.distances[indices]**2))


def consistent_first_rows(dim, n, ds, lr):

	for a in ds:
		H = np.zeros((dim, dim)).astype(np.int)
		H[0, 0] = a
		if is_consistent(dim, H, lr) >= 0:
			yield a


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

					H = np.zeros((dim, dim)).astype(np.int)
					H[0] = [a, 0, 0]
					H[1] = [s, b, 0]
					if is_consistent(dim, H, lr) < 0:
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

		print(it, r, H.reshape(-1), rmsd)

		group_index = n**dim // np.prod(np.diag(H))
		data.append((rmsd, group_index, H))

	print(n, len(data), r, it)
	return data, lr
