from ase.build import molecule
from ase.collections import g2
from ase.text import plot

for atoms in g2:
    print()
    print(plot(atoms))
    print()
