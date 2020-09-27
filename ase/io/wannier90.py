import numpy as np

from ase import Atoms


def read_wout_all(fileobj):
    lines = fileobj.readlines()

    for n, line in enumerate(lines):
        if line.strip().lower().startswith('lattice vectors (ang)'):
            break
    else:
        raise ValueError('Could not fine lattice vectors')

    cell = [[float(x) for x in line.split()[-3:]]
            for line in lines[n + 1:n + 4]]

    for n, line in enumerate(lines):
        if 'cartesian coordinate (ang)' in line.lower():
            break
    else:
        raise ValueError('Could not find coordinates')

    positions = []
    symbols = []
    n += 2
    while True:
        words = lines[n].split()
        if len(words) == 1:
            break
        positions.append([float(x) for x in words[-4:-1]])
        symbols.append(words[1])
        n += 1

    atoms = Atoms(symbols, positions, cell=cell)

    n = len(lines) - 1
    while n > 0:
        if lines[n].strip().lower().startswith('final state'):
            break
        n -= 1
    else:
        return atoms, np.zeros((0, 3)), np.zeros((0,))

    n += 1
    centers = []
    widths = []
    while True:
        line = lines[n].strip()
        if line.startswith('WF'):
            centers.append([float(x)
                            for x in
                            line.split('(')[1].split(')')[0].split(',')])
            widths.append(float(line.split()[-1]))
            n += 1
        else:
            break

    return atoms, np.array(centers), np.array(widths)


def read_wout(fileobj, include_wannier_function_centers=True):
    atoms, centers, widths = read_wout_all(fileobj)
    if include_wannier_function_centers:
        atoms += Atoms(f'X{len(centers)}', centers)
    return atoms
