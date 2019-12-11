#2 non-calculator files
from ase.build import fcc100
from ase.io import write

slab = fcc100('Al', size=(2, 2, 3))
write('slab1.cif',slab)
from numpy.random import random
slab.positions+=random((len(slab),3))
write('slab2.cif',slab)

from ase.test import cli
cli('ase diff slab1.cif slab2.cif')
