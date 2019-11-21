from ase.build import molecule
from ase.text import plot

atoms = molecule('CH3CH2OH')

txt = plot(atoms)

print(txt)
