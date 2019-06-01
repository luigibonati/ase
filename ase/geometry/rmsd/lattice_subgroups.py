import itertools
import numpy as np

from ase.geometry.rmsd.cell_projection import intermediate_representation
from ase.geometry.rmsd.standard_form import standardize_atoms
from ase.geometry.rmsd.alignment import LatticeComparator


class LatticeReducer:

    def __init__(self, atoms):

        res = standardize_atoms(atoms.copy(), atoms.copy(), False)
        a, b, atomic_perms, axis_perm = res
        pa, pb, _, _ = intermediate_representation(a, b, 'central')

        lc = LatticeComparator(pa, pb)
        dim = lc.dim

        num_atoms = len(lc.numbers)
        nx, ny, nz = len(lc.xindices), len(lc.yindices), len(lc.zindices)
        if dim == 1:
            distances = np.zeros(nz)
            permutations = -np.ones((nz, num_atoms)).astype(np.int)
        elif dim == 2:
            distances = np.zeros((nx, ny))
            permutations = -np.ones((nx, ny, num_atoms)).astype(np.int)
        elif dim == 3:
            distances = np.zeros((nx, ny, nz))
            permutations = -np.ones((nx, ny, nz, num_atoms)).astype(np.int)

        self.lc = lc
        self.cindices = [lc.xindices, lc.yindices, lc.zindices]
        self.distances = distances
        self.permutations = permutations

    def get_point(self, c):

        c = tuple(c)
        if self.permutations[c][0] != -1:
            return self.permutations[c]

        rmsd, permutation = self.lc.cherry_pick(c)
        self.distances[c] = rmsd
        self.permutations[c] = permutation
        return permutation


def invert_permutation(p):
    return np.argsort(p)


def get_divisors(n):
    return [i for i in range(1, n + 1) if n % i == 0]


def get_group_elements(n, dim, H):

    assert dim in [1, 2, 3]

    size = 1
    for e in np.diag(H):
        if e != 0:
            size *= n // e

    indices = np.zeros((size, dim)).astype(np.int)
    indices[:, 0] = H[0, 0] * np.arange(size)

    if dim >= 2 and H[1, 1] != 0:
        k = n // H[0, 0]
        r = np.arange(size) // k
        indices[:, 0] += H[1, 0] * r
        indices[:, 1] += H[1, 1] * r

    if dim == 3 and H[2, 2] != 0:
        k = n**2 // (H[1, 1] * H[0, 0])
        r = np.arange(size) // k
        indices[:, 0] += H[2, 0] * r
        indices[:, 1] += H[2, 1] * r
        indices[:, 2] += H[2, 2] * r

    return indices % n


def permutationally_consistent(dim, H, lr):

    if lr is None:  # Used for unit tests only
        return 0

    n = len(lr.lc.s0)
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
        if permutationally_consistent(dim, H, lr) >= 0:
            yield a


def solve_linear_congruence(n, a, b, c, s, v):

    for u in range(a):
        if (n // c * u) % a == (n * v * s // (b * c)) % a:
            return u
    raise Exception("u not found")


def get_basis(dim, n, lr):
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
                    if permutationally_consistent(dim, H, lr) < 0:
                        continue

                    for w in range(B * gcd(t, X) // X):

                        v = b * X * w // (B * gcd(t, X))
                        u0 = solve_linear_congruence(n, a, b, c, s, v)

                        for z in range(C):
                            u = u0 + a * z // C
                            yield np.array([[a, 0, 0], [s, b, 0], [u, v, c]])


def number_of_subgroups(dim, n):

    def _P(n):
        gcd = np.gcd
        return sum([gcd(k, n) for k in range(1, n + 1)])

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


def find_consistent_reductions(atoms):

    n = len(atoms)
    dim = sum(atoms.pbc)
    lr = LatticeReducer(atoms)

    it = 0
    data = []
    for H in get_basis(dim, n, lr):

        it += 1
        rmsd = permutationally_consistent(dim, H, lr)
        if rmsd < 0:
            continue

        # print(it, H.reshape(-1), rmsd)
        group_index = n**dim // np.prod(np.diag(H))
        data.append((rmsd, group_index, H))

    # print(n, len(data), it)
    return data, lr
