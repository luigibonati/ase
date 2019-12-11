#1 traj file
from ase.build import fcc100, add_adsorbate
from ase.constraints import FixAtoms, FixedPlane
from ase.calculators.emt import EMT
from ase.optimize import QuasiNewton

slab = fcc100('Al', size=(2, 2, 3))
add_adsorbate(slab, 'Au', 1.7, 'hollow')
slab.center(axis=2, vacuum=4.0)
mask = [atom.tag > 1 for atom in slab]
fixlayers = FixAtoms(mask=mask)
plane = FixedPlane(-1, (1, 0, 0))
slab.set_constraint([fixlayers, plane])
slab.set_calculator(EMT())
qn = QuasiNewton(slab, trajectory='mep0.traj')
qn.run(fmax=0.02)

from ase.test import cli
cli('ase diff mep4.traj -c')
