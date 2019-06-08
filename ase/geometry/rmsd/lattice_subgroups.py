import itertools
import numpy as np


class DummyReducer:
    def permutationally_consistent(*args):
        return True


def get_divisors(n):
    return [i for i in range(1, n + 1) if n % i == 0]


def get_group_elements(dims, H):

    dim = len(dims)
    assert dim in [1, 2, 3]

    if dim == 1:
        m = dims[0]
    elif dim == 2:
        m, n = dims
    else:
        m, n, r = dims

    size = 1
    for e, _n in zip(np.diag(H), dims):
        if e != 0:
            size *= _n // e

    indices = np.zeros((size, dim)).astype(np.int)
    indices[:, 0] = H[0, 0] * np.arange(size)

    if dim >= 2 and H[1, 1] != 0:
        k = m // H[0, 0]
        _r = np.arange(size) // k
        indices[:, 0] += H[1, 0] * _r
        indices[:, 1] += H[1, 1] * _r

    if dim == 3 and H[2, 2] != 0:
        k = m * n // (H[1, 1] * H[0, 0])
        _r = np.arange(size) // k
        indices[:, 0] += H[2, 0] * _r
        indices[:, 1] += H[2, 1] * _r
        indices[:, 2] += H[2, 2] * _r

    for i, e in enumerate(dims):
        indices[:, i] %= e
    return indices


def consistent_first_rows(dim, dm, lr):

    for a in dm:
        H = np.zeros((dim, dim)).astype(np.int)
        H[0, 0] = a
        if lr.permutationally_consistent(H) >= 0:
            yield a


def solve_linear_congruence(r, a, b, c, s, v):

    for u in range(a + 1):
        if (r // c * u) % a == (r * v * s // (b * c)) % a:
            return u

    raise Exception("u not found")


def get_basis(dims, lr=DummyReducer):
    """
    The subgroup theory used here is described for two-dimensional lattices in:

        Representing and counting the subgroups of the group Z_m x Z_n
        Mario Hampejs, Nicki Holighaus, László Tóth, and Christoph Wiesmeyr
        Journal of Numbers, vol. 2014, Article ID 491428
        http://dx.doi.org./10.1155/2014/491428
        https://arxiv.org/abs/1211.1797

    and for three-dimensional lattices in:

        On the subgroups of finite Abelian groups of rank three
        Mario Hampejs and László Tóth
        Annales Univ. Sci. Budapest., Sect. Comp. 39 (2013), 111–124
        https://arxiv.org/abs/1304.2961
    """

    dim = len(dims)
    assert dim in [1, 2, 3]
    gcd = np.gcd

    if dim == 1:
        m = dims[0]
    elif dim == 2:
        m, n = dims
    else:
        m, n, r = dims

    dm = get_divisors(m)

    if dim == 1:
        for d in consistent_first_rows(dim, dm, lr):
            yield np.array([[d]])

    elif dim == 2:

        dn = get_divisors(n)

        for a in consistent_first_rows(dim, dm, lr):
            for b in dn:
                for t in range(gcd(a, n // b)):
                    s = t * a // gcd(a, n // b)
                    yield np.array([[a, 0], [s, b]])

    elif dim == 3:

        dn = get_divisors(n)
        dr = get_divisors(r)

        for a in consistent_first_rows(dim, dm, lr):

            for b, c in itertools.product(dn, dr):

                A = gcd(a, n // b)
                B = gcd(b, r // c)
                C = gcd(a, r // c)
                ABC = A * B * C
                X = ABC // gcd(a * r // c, ABC)

                for t in range(A):

                    s = a * t // A

                    H = np.zeros((dim, dim)).astype(np.int)
                    H[0] = [a, 0, 0]
                    H[1] = [s, b, 0]
                    if lr.permutationally_consistent(H) < 0:
                        continue

                    for w in range(B * gcd(t, X) // X):

                        v = b * X * w // (B * gcd(t, X))
                        u0 = solve_linear_congruence(r, a, b, c, s, v)

                        for z in range(C):
                            u = u0 + a * z // C
                            yield np.array([[a, 0, 0], [s, b, 0], [u, v, c]])


def number_of_subgroups(dims):

    def _P(n):
        gcd = np.gcd
        return sum([gcd(k, n) for k in range(1, n + 1)])

    dim = len(dims)
    assert dim in [1, 2, 3]
    gcd = np.gcd

    if dim == 1:
        m = dims[0]
    elif dim == 2:
        m, n = dims
    else:
        m, n, r = dims

    dm = get_divisors(m)

    if dim == 1:
        return len(dm)
    elif dim == 2:
        dn = get_divisors(n)
        return sum([gcd(a, b) for a in dm for b in dn])
    else:
        dn = get_divisors(n)
        dr = get_divisors(r)

        total = 0
        for a, b, c in itertools.product(dm, dn, dr):
            A = gcd(a, n // b)
            B = gcd(b, r // c)
            C = gcd(a, r // c)
            ABC = A * B * C

            X = ABC // gcd(a * r // c, ABC)
            total += ABC // X**2 * _P(X)
        return total
