#2 non-calculator files
from ase.build import fcc100
from ase.io import write

slab = fcc100('Al', size=(2, 2, 3))
write('slab.cif',[slab,slab])

from ase.test import cli
cli('ase diff slab.cif')
