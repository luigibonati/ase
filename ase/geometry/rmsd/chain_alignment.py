import heapq
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


def calculate_lb_cost(P, Q, dtheta, interval):

    dtheta = np.copy(dtheta)
    t0, t1 = interval

    dt0 = (dtheta - t0) % (2 * np.pi)
    dt1 = (dtheta - t1) % (2 * np.pi)
    db0 = np.minimum(np.abs(dt0), np.abs(2 * np.pi - dt0))
    db1 = np.minimum(np.abs(dt1), np.abs(2 * np.pi - dt1))

    boundary = np.full_like(dtheta, t0)
    indices = np.where(db1 < db0)
    boundary[indices] = t1

    indices = np.where((dtheta < t0) | (dtheta > t1))
    dtheta[indices] = boundary[indices]

    assert (dtheta >= t0).all()
    assert (dtheta <= t1).all()

    sint = np.sin(dtheta)
    cost = np.cos(dtheta)

    px = (cost.T * P[:, 0] - sint.T * P[:, 1]).T
    py = (sint.T * P[:, 0] + cost.T * P[:, 1]).T
    return (px - Q[:, 0])**2 + (py - Q[:, 1])**2


def concatenate_permutations(perms):

    perm = []
    for p in perms:
        perm.extend(list(p + len(perm)))
    return np.array(perm)


def limiting_permutation(P, Q, eindices, dzsqs, dthetas, theta):

    U = rotation_matrix(np.sin(theta), np.cos(theta))

    perms = []
    for indices, dzsq in zip(eindices, dzsqs):

        p = P[indices, :2]
        rq = np.dot(Q[indices, :2], U.T)

        cost = cdist(p, rq, metric='sqeuclidean') + dzsq
        perm = linear_sum_assignment(cost)
        perms.append(perm[1])

    return concatenate_permutations(perms)


def _register(P, Q, eindices, cell_length, best):

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

        if abs(a - b) < np.deg2rad(1E-12):
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
            if a in cached_limits:
                perma = cached_limits[a]
            else:
                perma = limiting_permutation(P, Q, eindices, dzsqs, dthetas, a)
                cached_limits[a] = perma

            if b in cached_limits:
                perma = cached_limits[b]
            else:
                permb = limiting_permutation(P, Q, eindices, dzsqs, dthetas, b)
                cached_limits[b] = permb

            if (perma == permb).all():
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
    if best is None or 1:    # turn off for now
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
