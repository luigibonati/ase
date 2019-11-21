from ase.units import Bohr
from ase.data import chemical_symbols
import numpy as np


_colors = {'blue': '0;34',
           'light red': '1;31',
           'light purple': '1;35',
           'brown': '0;33',
           'purple': '0;35',
           'yellow': '1;33',
           'dark gray': '1;30',
           'light cyan': '1;36',
           'black': '0;30',
           'light green': '1;32',
           'cyan': '0;36',
           'green': '0;32',
           'light blue': '1;34',
           'light gray': '0;37',
           'white': '1;37',
           'red': '0;31',
           'old': '1;31;41',  # To do: proper names, reorganize
           'new': '1;33;42',  # These are used by gtprevmsgdiff
           None: None}


def _ansiwrap(string, id):
    if id is None:
        return string
    tokens = []
    for line in string.split('\n'):
        if len(line) > 0:
            line = '\x1b[%sm%s\x1b[0m' % (id, line)
        tokens.append(line)
    return '\n'.join(tokens)


class ANSIColors:
    def get(self, name):
        color = _colors[name.replace('_', ' ')]

        def colorize(string):
            return _ansiwrap(string, color)
        return colorize

    __getitem__ = get
    __getattr__ = get


ansi_nocolor = '\x1b[0m'
ansi = ANSIColors()



def plot(atoms):
    """Ascii-art plot of the atoms."""

#   y
#   |
#   .-- x
#  /
# z

    cell_cv = atoms.get_cell()
    if not atoms.cell or (cell_cv - np.diag(cell_cv.diagonal())).any():
        atoms = atoms.copy()
        atoms.cell = [1, 1, 1]
        atoms.center(vacuum=2.0)
        cell_cv = atoms.get_cell()
        plot_box = False
    else:
        plot_box = True

    cell = np.diagonal(cell_cv) / Bohr
    positions = atoms.get_positions() / Bohr
    numbers = atoms.get_atomic_numbers()

    s = 1.3
    nx, ny, nz = n = (s * cell * (1.0, 0.25, 0.5) + 0.5).astype(int)
    sx, sy, sz = n / cell
    grid = Grid(nx + ny + 4, nz + ny + 1)
    positions = (positions % cell + cell) % cell
    ij = np.dot(positions, [(sx, 0), (sy, sy), (0, sz)])
    ij = np.around(ij).astype(int)
    for a, Z in enumerate(numbers):
        symbol = chemical_symbols[Z]
        i, j = ij[a]
        depth = positions[a, 1]
        for n, c in enumerate(symbol):
            grid.put(c, i + n + 1, j, depth)
    if plot_box:
        k = 0
        for i, j in [(1, 0), (1 + nx, 0)]:
            grid.put('*', i, j)
            grid.put('.', i + ny, j + ny)
            if k == 0:
                grid.put('*', i, j + nz)
            grid.put('.', i + ny, j + nz + ny)
            for y in range(1, ny):
                grid.put('/', i + y, j + y, y / sy)
                if k == 0:
                    grid.put('/', i + y, j + y + nz, y / sy)
            for z in range(1, nz):
                if k == 0:
                    grid.put('|', i, j + z)
                grid.put('|', i + ny, j + z + ny)
            k = 1
        for i, j in [(1, 0), (1, nz)]:
            for x in range(1, nx):
                if k == 1:
                    grid.put('-', i + x, j)
                grid.put('-', i + x + ny, j + ny)
            k = 0
    return '\n'.join([''.join([chr(x) for x in line])
                      for line in np.transpose(grid.grid)[::-1]])


class Grid:
    def __init__(self, i, j):
        self.grid = np.zeros((i, j), np.int8)
        self.grid[:] = ord(' ')
        self.depth = np.zeros((i, j))
        self.depth[:] = 1e10

    def put(self, c, i, j, depth=1e9):
        if depth < self.depth[i, j]:
            self.grid[i, j] = ord(c)
            self.depth[i, j] = depth
