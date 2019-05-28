import itertools
import numpy as np


def gauss(B, hu, hv):

    u = np.dot(B.T, hu)
    v = np.dot(B.T, hv)

    max_it = 100000    # in practice this is not exceeded
    for it in range(max_it):

        x = int(round(np.dot(u, v) / np.dot(u, u)))

        hu, hv = hv - x * hu, hu

        u = np.dot(B.T, hu)
        v = np.dot(B.T, hv)
        if np.dot(u, u) >= np.dot(v, v):
            return hv, hu

    raise Exception("Gaussian basis not found after %d iterations" % max_it)


def relevant_vectors_2D(u, v):

    cs = np.array([e for e in itertools.product([-1, 0, 1], repeat=2)])
    vs = np.dot(cs, [u, v])
    indices = np.argsort(np.linalg.norm(vs, axis=1))[:7]
    return vs[indices], cs[indices]


def closest_vector(t0, u, v):

    t = t0[::]
    rs, cs = relevant_vectors_2D(u, v)
    a = np.array([0, 0])

    dprev = float("inf")
    max_it = 100000    # in practice this is not exceeded
    for it in range(max_it):

        ds = np.linalg.norm(rs + t, axis=1)
        index = np.argmin(ds)
        if index == 0 or ds[index] >= dprev:
            return a

        dprev = ds[index]
        r = rs[index]
        kopt = int(round(-np.dot(t, r) / np.dot(r, r)))
        a += kopt * cs[index]
        t = t0 + a[0] * u + a[1] * v

    raise Exception("Closest vector not found after %d iterations" % max_it)


def reduce_basis(B):

    # calculates a minkowski basis and a reduction path

    path = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    norms = np.linalg.norm(np.dot(path, B), axis=1)

    max_it = 100000    # in practice this is not exceeded
    for it in range(max_it):

        indices = np.argsort(norms)
        path = path[indices]

        hw = path[2]
        hu, hv = gauss(B, path[0], path[1])
        path[0] = hu
        path[1] = hv

        u = np.dot(B.T, hu)
        v = np.dot(B.T, hv)
        w = np.dot(B.T, hw)
        Bprime = np.array([u, v, w])

        X = u / np.linalg.norm(u)
        Y = v - X * np.dot(v, X)
        Y /= np.linalg.norm(Y)

        pu, pv, pw = np.dot(Bprime, np.array([X, Y]).T)
        nb = closest_vector(pw, pu, pv)
        hw = np.dot([nb[0], nb[1], 1], [hu, hv, hw])

        path = np.array([hu, hv, hw])
        Bprime = np.dot(path, B)

        norms = np.diag(np.dot(Bprime, Bprime.T))
        if norms[2] >= norms[1] or (nb == 0).all():
            return Bprime, path

    raise Exception("Minkowski basis not found after %d iterations" % max_it)
