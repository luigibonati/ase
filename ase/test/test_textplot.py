from ase.collections import g2
from ase.text import plot


def test_textplot():
    for atoms in g2:
        print()
        print(plot(atoms))
        print()
