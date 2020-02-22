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
atom_colors = [None] * 119


def set_colors(symbols, color):
    from ase.symbols import Symbols
    for Z in Symbols.fromsymbols(symbols).numbers:
        atom_colors[Z] = color


table = """\

  H                                                 He
 Li Be                                B  C  N  O  F Ne
 Na Mg                               Al Si  P  S Cl Ar
  K Ca Sc Ti  V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr
 Rb Sr  Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te  I Xe
 Cs Ba  * Hf Ta  W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn
 Fr Ra ** Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og

      * La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu
     ** Ac Th Pa  U Np Pu Am Cm Bk Cf Es Fm Md No Lr
"""

col = set_colors

# We choose colours based more or less on CPK.

col('X', 'red')
col('H', 'light gray')
col('LiNaKRbCsFr', 'purple')  # Group 1
col('BeMgCaSr', 'light green')  # Group 2
col('BaRa', 'green')  # More group 2
col('HeNeAr', 'light cyan')  # Nobles part 1
col('KrXeRn', 'cyan')  # Nobles part 2

# Transition metals 3d
col('ScTiV', 'light gray')
col('Cr', 'cyan')
col('Mn', 'purple')
col('Fe', 'light red')
col('Co', 'light purple')
col('Ni', 'light green')
col('Cu', 'brown')
col('Zn', 'light blue')

# 4d
col('YZrNb', 'light cyan')
col('MoTcRuRhPd', 'cyan')
col('Ag', 'white')
col('Cd', 'light purple')

# 5d
col('LaHfTaW', 'light blue')
col('ReOsIr', 'blue')
col('PtHg', 'light gray')
col('Au', 'yellow')

# The main-group elements are a bit of a colour hodge-podge
col('BSi', 'light purple')
col('CPb', 'dark gray')
col('Al', 'light gray')
col('GaInTlPSeTePoAt', 'brown')
col('GeSn', 'cyan')
col('N', 'light blue')
col('O', 'light red')
col('FCl', 'light green')
col('Br', 'red')
col('S', 'yellow')
col('AsSbBiI', 'purple')

# Lanthanides
col('Ce', 'white')
col('PrNdPmSmEuGdTbDy', 'light cyan')
col('HoErTm', 'light green')
col('YbLu', 'green')

# Actinides and Ac
col('AcThPaU', 'light blue')
col('NpPuAm', 'blue')
col('CmBkCfEsFmMdNoLr', 'purple')

# Post-actinides
col('RfDbSgBhHsMt', 'red')
col('DsRgCnNhFlMcLvTsOg', 'white')


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
        atoms.center(vacuum=0.0)
        cell_cv = atoms.get_cell().complete()
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
    ij = np.dot(positions, [(sx, 0), (sy, sy), (0, sz)])
    ij = np.around(ij).astype(int)
    for a, Z in enumerate(numbers):
        symbol = chemical_symbols[Z]
        i, j = ij[a]
        depth = positions[a, 1]
        for n, c in enumerate(symbol):
            grid.put(c, i + n + 1, j, depth, Z)
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

    lines = []

    chargrid = grid.grid.T[::-1]
    colorgrid = grid.color_key.T[::-1]

    for line, linecolors in zip(chargrid, colorgrid):
        tokens = []
        for x, color_key in zip(line, linecolors):
            txt = chr(x)
            if color_key != -1:
                color = atom_colors[color_key]
                txt = ansi[color](txt)
            tokens.append(txt)
        line = ''.join(tokens).rstrip()
        lines.append(line)

    # We use \r for the benefit of curses
    return '\r\n'.join(lines)


class Grid:
    def __init__(self, i, j):
        self.grid = np.zeros((i, j), np.int8)
        self.grid[:] = ord(' ')
        self.depth = np.zeros((i, j))
        self.depth[:] = 1e10
        self.color_key = -np.ones((i, j), int)

    def put(self, c, i, j, depth=1e9, color_key=-1):
        if depth < self.depth[i, j]:
            self.grid[i, j] = ord(c)
            self.depth[i, j] = depth
            self.color_key[i, j] = color_key


def main():
    # main(stdscr)
    import sys
    from ase.io import iread

    if len(sys.argv) == 1:
        from ase.data import atomic_numbers
        import re

        def substitute(match):
            sym = match.group()
            Z = atomic_numbers[sym]
            return ansi[atom_colors[Z]](sym)

        print(re.sub(r'[A-Z][a-z]?', substitute, table))
        return

    for fname in sys.argv[1:]:
        for i, atoms in enumerate(iread(fname)):
            print('{}@{}: {}'.format(fname, i, atoms))
            txt = plot(atoms)
            print(txt)
            print()

    """
        for i in range(200):
            stdscr.clear()
            atoms = atoms0.copy()
            atoms.rotate(i, 'z', center='com')
            txt = plot(atoms)
            stdscr.addstr(0, 0, 'hello {}'.format(ansi.red(str(i))))
            curses.doupdate()
            #print(ansi.red('hello world'))
            time.sleep(0.02)
            stdscr.noutrefresh()
            #stdscr.refresh()
    """


if __name__ == '__main__':
    main()
    # import curses
    # curses.wrapper(main)
