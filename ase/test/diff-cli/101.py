#2 non-calculator files
from ase.build import fcc100
from ase.io import write

slab = fcc100('Al', size=(2, 2, 3))
slab2 = slab.copy()
slab2.positions += [0,0,1]
write('slab.cif',[slab,slab2])

from ase.test import cli
cli('ase diff slab.cif')
