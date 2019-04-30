import numpy as np
import itertools
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


def optimal_permutation(P, Q, theta):

	U = rotation_matrix(np.sin(theta), np.cos(theta))
	RP = np.dot(P[:, :2], U.T)

	linear_sum_assignment

	nrmsdsq = np.sum((RP - Q[:, :2])**2)
	return nrmsdsq


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


def limiting_permutation(P, Q, eindices, dzsqs, dthetas, theta):

	U = rotation_matrix(np.sin(theta), np.cos(theta))

	obj = 0
	perms = []
	for indices, dzsq in zip(eindices, dzsqs):

		p = P[indices, :2]
		rq = np.dot(Q[indices, :2], U.T)

		cost = cdist(p, rq, metric='sqeuclidean') + dzsq
		perm = linear_sum_assignment(cost)
		obj += np.sum(cost[perm])
		perms.append(perm[1])

	#print("obj:", obj)
	perm = concatenate_permutations(perms)
	return perm

def get_cost_matrix(us, vs):

	n = len(us)
	Cx = np.zeros((n,n))
	Cy = np.zeros((n,n))
	for i, u in enumerate(us):
		for j, v in enumerate(vs):
			Cx[i, j] = np.linalg.det([u, v])
			Cy[i, j] = np.dot(u, v)
	return Cx, Cy


def get_plane(points, s):

	midpoint = np.mean(points, axis=0)
	ps = points[s]
	u = ps[1] - ps[0]
	v = ps[2] - ps[0]
	plane = np.cross(u, v).astype(np.double)
	if np.dot(ps[0] - midpoint, plane) < 0:
		plane = -plane
	plane /= np.linalg.norm(plane)
	return plane


def get_unique_points(ps):

	unique = [ps[0]]

	for p in ps[1:]:
		if all([np.linalg.norm(e - p) > 1E-6 for e in unique]):
			unique += [p]
	return np.array(unique)


def optimize(eindices, Cs, sint, cost):

	point = np.zeros(3)
	perms = []
	for indices, C in zip(eindices, Cs):

		Cx, Cy, Cz = C
		w = sint * Cx + cost * Cy + Cz

		perm = linear_sum_assignment(-w)
		perms.append(perm[1])
		px = np.sum(Cx[perm])
		py = np.sum(Cy[perm])
		pz = np.sum(Cz[perm])
		point += [px, py, pz]

	return point, concatenate_permutations(perms)


def brute_force(P, Q, eindices, cell_length):

	Cz = cdist(P[:, 2:], Q[:, 2:], metric='sqeuclidean')
	Cz = np.minimum(Cz, cdist(P[:, 2:] - cell_length, Q[:, 2:], metric='sqeuclidean'))
	Cz = np.minimum(Cz, cdist(P[:, 2:] + cell_length, Q[:, 2:], metric='sqeuclidean'))

	gperm = [itertools.permutations(range(len(e))) for e in eindices]
	perms = itertools.product(*gperm)

	best = (float("inf"), None, None)
	n = len(P)
	for p in perms:
		p = [np.array(e) for e in p]
		perm = concatenate_permutations(p)
		obj, U = calculate_actual_cost(P, Q[perm])
		obj += np.sum(Cz[np.arange(n), perm])
		best = min(best, (obj, perm, U), key=lambda x: x[0])
	return best


def _register(P, Q, eindices, cell_length, best):

	n = len(P)
	if n <= 4:
		return brute_force(P, Q, eindices, cell_length)

	Cs = []
	for indices in eindices:

		Cx, Cy = get_cost_matrix(P[indices, :2], Q[indices, :2])

		Cz = np.dot(P[indices, 2:], Q[indices, 2:].T)
		Cz = np.minimum(Cz, np.dot(P[indices, 2:] - cell_length, Q[indices, 2:].T))
		Cz = np.minimum(Cz, np.dot(P[indices, 2:] + cell_length, Q[indices, 2:].T))

		Cs.append((Cx, Cy, Cz))

	p0 = np.zeros(3)
	for Cx, Cy, Cz in Cs:
		p0 += [np.trace(Cx), np.trace(Cy), np.trace(Cz)]

	print(P)
	print(Q)

	ps = np.array([p0])
	perms = np.array([np.arange(n)])

	ps = np.array([[0, 0, -1.]])
	perms = np.array([-np.arange(n)])
	while 1:
		theta = np.random.uniform(0, 2 * np.pi, 1)
		p, perm = optimize(eindices, Cs, np.sin(theta), np.cos(theta))
		ps = np.concatenate((ps, [p]))
		perms = np.concatenate((perms, [perm]))
		print(ps)
		print(perms)
		print(np.linalg.matrix_rank(ps - ps[0]))
		print()
		if np.linalg.matrix_rank(ps - ps[0]) >= 3:
			break
		if len(ps) > 4: break

	asdf

	ps = np.array(ps)
	simplices = scipy.spatial.ConvexHull(ps).simplices
	extreme = np.unique(simplices)
	ps = ps[extreme]
	perms = perms[extreme]



	points = np.array(initial)
	finished = np.array([[0, 0, 0]])

	extreme = np.unique(scipy.spatial.ConvexHull(points).simplices)
	points = points[extreme]
	print(points)

	while 1:
		simplices = np.sort(scipy.spatial.ConvexHull(points).simplices)
		planes = [get_plane(points, s) for s in simplices]
		todo = np.array([p for p in planes if all([np.linalg.norm(e - p) > 1E-6 for e in finished])])
		todo = get_unique_points(todo)
		print("num. todo:", len(todo))
		print(todo)
		if len(todo) == 0:
			break

		indices = np.argsort(np.abs(todo[:,2]))[::]
		#indices = np.argsort(todo[:,2])[::]
		#random.shuffle(indices)
		todo = todo[indices]
		plane = todo[0]
		print(plane)
		print("maxval:", np.max(np.dot(points, plane.T)))

		kgb = plane[1]
		kc = plane[2]

		finished = np.concatenate((finished, [plane]))

		trial = np.array([int(round(v.x)) for v in vs])
		print("trial:", trial)

		if all([np.linalg.norm(e - trial) > 1E-6 for e in points]):
			points = np.concatenate((points, [trial]))
			extreme = np.unique(scipy.spatial.ConvexHull(points).simplices)
			new = extreme[-1] == len(points) - 1
			points = points[extreme]
			print("new points:\n", points)

	print("points:")
	print(points)


def __register(P, Q, eindices, cell_length, best):

	dzsqs = []
	dthetas = []
	for indices in eindices:

		p = P[indices, 2:]
		q = Q[indices, 2:]
		dzsq = cdist(p, q, metric='sqeuclidean')
		for i in [-1, 1]:
			d = cdist(p + i * cell_length, q, metric='sqeuclidean')
			dzsq = np.minimum(dzsq, d)
		dzsqs.append(dzsq)

		thetasP = np.arctan2(P[indices, 1], P[indices, 0])
		thetasQ = np.arctan2(Q[indices, 1], Q[indices, 0])
		dtheta = np.array([thetasQ - p for p in thetasP]) % (2 * np.pi)
		dthetas.append(dtheta)

	cached_limits = {}

	interval = (0, 2 * np.pi)
	queue = [(0, interval)]
	heapq.heapify(queue)
	while queue:

		lb, interval = heapq.heappop(queue)
		a, b = interval

		if lb > best[0]:
			continue

		if abs(a - b) < 1E-3:#np.deg2rad(1E-12):
			continue

		if abs(lb - best[0]) < 1E-12:
			continue

		dzsum = 0
		lbval = 0
		perms = []
		for indices, dzsq, dtheta in zip(eindices, dzsqs, dthetas):

			p = P[indices, :2]
			q = Q[indices, :2]

			lbcost = calculate_lb_cost(p, q, dtheta, interval) + dzsq
			perm = linear_sum_assignment(lbcost)
			perms.append(perm[1])
			_lbval = np.sum(lbcost[perm])
			lbval += _lbval

			dzsum += np.sum(dzsq[perm])

		assert lbval >= lb - 1E-6
		perm = concatenate_permutations(perms)

		estimate, U = calculate_actual_cost(P, Q[perm]) + dzsum
		if estimate <= best[0]:
			best = (estimate, perm, U)


		if abs(b - a) < np.deg2rad(100):

			perms = []
			for t in [a, b]:
				if t in cached_limits:
					permt = cached_limits[t]
				else:
					permt = limiting_permutation(P, Q, eindices, dzsqs, dthetas, t)
					cached_limits[t] = permt

				estimate, U = calculate_nrmsdsq(P, Q[permt], cell_length)
				if estimate <= best[0]:
					best = (estimate, permt, U)

				perms.append(permt)

			if (perms[0] == perms[1]).all():
				continue

		a, b = interval
		interval0 = (a, (a + b) / 2)
		interval1 = ((a + b) / 2, b)

		heapq.heappush(queue, (lbval, interval0))
		heapq.heappush(queue, (lbval, interval1))

	obj, perm, U = best
	return obj, perm, U


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
