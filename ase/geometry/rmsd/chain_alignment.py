import numpy as np
from scipy.spatial.distance import cdist
from ase.geometry.rmsd.assignment import linear_sum_assignment


def rotation_matrix(sint, cost):

	norm = np.linalg.norm([sint, cost])
	if norm < 1E-10:
		return np.eye(2)

	U = np.array([[cost, -sint], [sint, cost]])
	return U / norm


def optimal_rotation(P, Q):

	assert P.shape[1] == 2
	assert Q.shape[1] == 2

	A = np.dot(P.T, Q)
	sint = A[0, 1] - A[1, 0]
	cost = A[0, 0] + A[1, 1]
	return rotation_matrix(sint, cost)


def calculate_actual_cost(P, Q):

	U = optimal_rotation(P[:, :2], Q[:, :2])
	RP = np.dot(P[:, :2], U.T)

	nrmsdsq = np.sum((RP - Q[:, :2])**2)
	return nrmsdsq, U


# A helper function.  Used for tests.
def calculate_nrmsdsq(P, Q, cell_length):

	dz = P[:, 2] - Q[:, 2]
	dzsq = np.min([(dz + i * cell_length)**2 for i in range(-1, 2)], axis=0)

	nrmsdsq, U = calculate_actual_cost(P[:, :2], Q[:, :2])
	return nrmsdsq + np.sum(dzsq), U


def concatenate_permutations(perms):

	perm = []
	for p in perms:
		perm.extend(list(p + len(perm)))
	return np.array(perm)


def get_cost_matrix(us, vs):

	vf = vs[:,[1,0]]
	vf[:,0] *= -1
	Cx = -2 * np.dot(us, vf.T)
	Cy = -2 * np.dot(us, vs.T)
	return Cx, Cy


def calculate_intercept(coeffs0, coeffs1):

	a, b, c = coeffs0 - coeffs1

	K = np.sqrt(a * a + b * b)
	if K < 1E-10:
		return None

	phi = np.arctan2(b, a)

	#sin(x + phi) = -c / K
	#y = x + phi

	y = np.arcsin(-c / K)
	return y - phi


#@profile
def optimize_and_calc(eindices, Cs, theta):

	sint = np.sin(theta)
	cost = np.cos(theta)

	perms = []
	c = np.zeros(3)
	for indices, C in zip(eindices, Cs):

		Cx, Cy, Cz = C
		w = sint * Cx + cost * Cy + Cz

		perm = linear_sum_assignment(w)
		perms.append(perm[1])
		dc = (np.sum(Cx[perm]), np.sum(Cy[perm]), np.sum(Cz[perm]))
		c += dc

	return perms, c


def minimum_angle(a, b):

	d = (b - a) % (2 * np.pi)
	return min(d, 2 * np.pi - d)


#@profile
def _register(P, Q, eindices, cell_length, best):

	Cs = []
	for indices in eindices:

		p = P[indices]
		q = Q[indices]
		Cx, Cy = get_cost_matrix(p[:, :2], q[:, :2])

		Cz = cdist(p[:, 2:], q[:, 2:], metric='sqeuclidean')
		Cz = np.minimum(Cz, cdist(p[:, 2:] - cell_length, q[:, 2:], metric='sqeuclidean'))
		Cz = np.minimum(Cz, cdist(p[:, 2:] + cell_length, q[:, 2:], metric='sqeuclidean'))
		Cz += np.sum(p[:, :2]**2) + np.sum(q[:, :2]**2)

		Cs.append((Cx, Cy, Cz))

	seen = set()
	pdict = {}
	coeffs = {}

	for theta in [0, np.pi]:
		perm, c = optimize_and_calc(eindices, Cs, theta)
		t = tuple(concatenate_permutations(perm))
		coeffs[t] = c
		seen.add(t)

		pdict[theta] = t
		if theta == 0:
			pdict[2 * np.pi] = t

	intervals = [(0, np.pi), (np.pi, 2 * np.pi)]


	if pdict[0] != pdict[np.pi]:
		while intervals:
			theta0, theta1 = intervals.pop(0)
			perm0, perm1 = pdict[theta0], pdict[theta1]

			intercept = calculate_intercept(coeffs[perm0], coeffs[perm1])
			if intercept is None:
				continue

			theta = intercept % (2 * np.pi)

			if minimum_angle(theta, theta0) < 1E-9: continue
			if minimum_angle(theta, theta1) < 1E-9: continue
			#if theta < theta0 or theta > theta1:
			#	print(theta0, theta1, theta)
			#	raise Exception("oh dear")

			perm, c = optimize_and_calc(eindices, Cs, theta)
			t = tuple(concatenate_permutations(perm))
			coeffs[t] = c

			if t not in seen:
				intervals.append((theta0, theta))
				intervals.append((theta, theta1))
				seen.add(t)
				pdict[theta] = t

	best = (float("inf"), None, None)
	for perm in seen:
		perm = np.array(perm)
		obj, U = calculate_nrmsdsq(P, Q[perm], cell_length)
		best = min(best, (obj, perm, U), key=lambda x: x[0])

	return best


def register_chain(P, Q, eindices, cell_length, best=None):

	n = len(P)
	if best is None or 1:
		permutation = np.arange(n)
		best = (float("inf"), permutation, np.eye(2))
	else:
		obj, permutation, U = best
		obj = obj**2 * n
		best = (obj, permutation, U)
	prev = best[0]

	best = _register(P, Q, eindices, cell_length, best)
	obj, permutation, U = best

	if obj != prev:
		rmsd = np.sqrt(obj / len(P))

		check, _ = calculate_nrmsdsq(P, Q[permutation], cell_length)
		assert abs(check - obj) < 1E-10
	else:
		rmsd = float("inf")

	return rmsd, permutation, U
