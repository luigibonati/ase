import numpy as np
import warnings


def permute_axes(atoms, permutation):

    scaled = atoms.get_scaled_positions()
    atoms.cell = atoms.cell[permutation]
    atoms.cell = atoms.cell[:, permutation]
    atoms.pbc = atoms.pbc[permutation]
    atoms.set_scaled_positions(scaled[:, permutation])


def standardize_axes(a, b):

    tol = 1E-12

    sa = np.sign(np.linalg.det(a.cell))
    sb = np.sign(np.linalg.det(b.cell))
    if sa != sb:
        warnings.warn('Cells have different handedness')

    # permute the axes such that the pbc's are ordered like this:
    #    1D: pbc = (0, 0, 1)
    #    2D: pbc = (1, 1, 0)
    #    3D: pbc = (1, 1, 1)

    pbc = a.pbc
    indices = [(tuple(np.roll(pbc, i)), i) for i in range(3)]
    indices = [e[1] for e in sorted(indices)]
    permutations = [np.roll(np.arange(3), i) for i in indices]

    dim = np.sum(pbc)
    if dim == 1 and tuple(pbc) != tuple(sorted(pbc)):
        permutation = permutations[0]
        assert tuple(pbc[permutation]) == (0, 0, 1)
    elif dim == 2 and tuple(pbc) != tuple(sorted(pbc, reverse=True)):
        permutation = permutations[-1]
        assert tuple(pbc[permutation]) == (1, 1, 0)
    else:
        permutation = np.arange(3)

    for atoms in [a, b]:
        lengths = np.linalg.norm(atoms.cell, axis=1)
        if min(lengths) < tol:
            raise Exception("Unit cell vectors may not have zero length")

    if dim <= 2:
        for atoms in [a, b]:
            permute_axes(atoms, permutation)

            # check that cell has appropriate structure (for chains and monolayers)
            bad_cell = False

            # want cell in format:
            # [X, X, 0]
            # [X, X, 0]
            # [0, 0, X]
            for i, j in [(0, 2), (1, 2), (2, 0), (2, 1)]:
                if abs(atoms.cell[i, j]) > tol:
                    bad_cell = True
                atoms.cell[i, j] = 0

            if bad_cell:
                warnings.warn('Bad cell: {}'.format(atoms.cell))

    return dim, a, b, permutation


def order_by_numbers(a, b, ignore_stoichiometry):

    assert len(a) == len(b)

    perms = []
    if ignore_stoichiometry:
        for atoms in [a, b]:
            n = len(atoms)
            atoms.numbers = np.ones(n).astype(np.int)
            perms.append(np.arange(n))
    else:
        # check stoichimetries are identical
        assert(sorted(a.numbers) == sorted(b.numbers))

        # sort atoms by atomic numbers
        for atoms in [a, b]:
            numbers = atoms.numbers
            indices = np.argsort(numbers)
            perms.append(indices)

            atoms.numbers = numbers[indices]
            atoms.set_positions(atoms.get_positions()[indices])
    return perms


def standardize_atoms(a, b, ignore_stoichiometry):
    """This function orders the atoms by z-number and permutes the coordinate
    axes into a standard form: the z-axis is periodic for 1D systems, the x and
    y-axes are perodic for 2D systems.  Standardization simplifies the code for
    finding the optimal alignment."""

    a = a.copy()
    b = b.copy()
    dim, a, b, permutation = standardize_axes(a, b)

    zperms = order_by_numbers(a, b, ignore_stoichiometry)
    return a, b, zperms, permutation
