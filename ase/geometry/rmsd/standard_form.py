import numpy as np
import warnings


def permute_axes(atoms, permutation):

    scaled = atoms.get_scaled_positions()
    atoms.cell = atoms.cell[permutation]
    atoms.cell = atoms.cell[:, permutation]
    atoms.pbc = atoms.pbc[permutation]
    atoms.set_scaled_positions(scaled[:, permutation])


def standardize_axes(a, b):

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

    if dim <= 2:
        for atoms in [a, b]:
            permute_axes(atoms, permutation)

            tol = 1E-10
            if (np.dot(atoms.cell[: 2], atoms.cell[2]) >= tol).any():
                if dim == 1:
                    raise Exception("Off-axis cell vectors not perpendicular to axis cell vector")
                elif dim == 2:
                    raise Exception("Out-of-plane cell vector not perpendicular to in-plane vectors")

            if dim == 1 and np.dot(atoms.cell[0], atoms.cell[1]) >= tol:
                    raise Exception("Off-axis cell vectors not perpendicular")

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
        if sorted(a.numbers) != sorted(b.numbers):
            raise Exception("Stoichiometries must be identical unless \
ignore_stoichiometry=True")

        # sort atoms by atomic numbers
        for atoms in [a, b]:
            numbers = atoms.numbers
            indices = np.argsort(numbers, kind='stable')
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
