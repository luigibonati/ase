import numpy as np
# -*- coding: utf-8 -*-
import sys

import numpy as np

from ase.geometry import find_mic
from ase.utils import basestring
from scipy.interpolate import interp1d

from ase.optimize.precon import Exp, C1, Pfrommer


from ase.calculators.calculator import Calculator, all_changes
from ase.build import bulk
from ase.optimize import QuasiNewton 
from ase.neb import NEB, NEBTools
import ase.io
from ase.io import read, write
from atomistica import Tersoff
from ase.optimize.ode import ODE12rOptimizer

strain=1.000

# Param settings
a = 3.5694
diamond = bulk('C', cubic = True, a = 3.561778)*3

#Set calculator
diamond.set_calculator(Tersoff())

Initial = diamond
pos = Initial.get_positions()
print(pos[0])

del Initial[0]
Initial.fix_all_cell = True
Initial.set_calculator(Tersoff())

Final = Initial.copy()

Final[0].position = pos[0]
Final.fix_all_cell = True
Final.set_calculator(Tersoff())

cell = diamond.get_cell()

Initial.set_cell(cell * strain, scale_atoms=True)
Final.set_cell(cell * strain, scale_atoms=True)

#Attach a writer
def initial(atoms=Initial):
    ase.io.write("initial.xyz", atoms, append=True)
def final(atoms=Final):
    ase.io.write("final.xyz", atoms, append=True)

# Initial state:
qn = ODE12rOptimizer(Initial)

qn.attach(initial)
qn.run(fmax=0.05)

# Final state:
qn = ODE12rOptimizer(Final)
qn.attach(final)
qn.run(fmax=0.05)



images = [Initial]
for image in range(5):
    image = Initial.copy()
    image.set_calculator(Tersoff())
    images.append(image)

images.append(Final)

neb = NEB(images, method='precon')
neb.interpolate()

opt = ODE12rOptimizer(neb)
opt.run(fmax=0.005)

nt = NEBTools(images)
fig = nt.plot_band()
fig.savefig('nebase.pdf')

images_copy = [img.copy() for img in images]
for image in images_copy:
    image.set_calculator(Tersoff())

f = neb.get_forces()

opt = ODE12rOptimizer(neb)
opt.run(fmax=0.05)

nt = NEBTools(images)
fig = nt.plot_band()
fig.savefig('neb.pdf')

#write('dump.xyz', neb.images)

#neb_s = NEB(images_copy, method='spline')
#neb_s.interpolate()

#f_s = neb_s.get_forces()

#print(abs(neb.get_positions() - neb_s.get_positions()).max())
#print(abs(f - f_s).max())

#opt = ODE12rOptimizer(neb_s)
#opt.run(fmax=0.05)

#nt = NEBTools(images)
#fig = nt.plot_band()
#fig.savefig('neb_s.pdf')

sys.exit(0)

#for image in images:
#    image.rattle(0.13)


nimages = len(images)
natoms = len(images[0])
s = np.zeros(nimages)
energies = np.zeros(nimages)

U = np.zeros((nimages,3*natoms))
U[0] = images[0].get_positions().reshape(3* natoms)


for i,img in enumerate(images[1:]):
    d = find_mic(img.get_positions() -
          images[0].get_positions(),
          images[0].get_cell(), images[0].pbc)[0]
    d_b = find_mic(img.get_positions() -
         images[-1].get_positions(),
         images[-1].get_cell(), images[-1].pbc)[0]
    tot_d = np.linalg.norm(find_mic(images[-1].get_positions() -
                                images[0].get_positions(),
                                images[0].get_cell(), images[0].pbc)[0])
    #print('this is d', d)
    #print('this is d', np.linalg.norm(d))
    #print('this is d_b', d_b)
    s[i+1] =  0.5*(np.linalg.norm(d)/tot_d + 1 - np.linalg.norm(d_b)/tot_d)
    print('this is s', s)
    d += images[0].get_positions()
    U[i+1] = d.reshape(3* natoms)
d_spline = interp1d(s, U.T, 'cubic')
dd_ds = d_spline._spline.derivative() # nab_d or d_tan


for i in range(0, nimages):
    energies[i] = images[i].get_potential_energy()
    
t1 = find_mic(images[1].get_positions() -
              images[0].get_positions(),
              images[0].get_cell(), images[0].pbc)[0]
nt1 = np.linalg.norm(t1)

for i in range(1, nimages - 1):
    t2 = find_mic(images[i + 1].get_positions() -
                  images[i].get_positions(),
                  images[i].get_cell(), images[i].pbc)[0]
    nt2 = np.linalg.norm(t2)

    # Tangents are improved according to formulas 8, 9, 10,
    # and 11 of paper I.
    if energies[i + 1] > energies[i] > energies[i - 1]:
        tangent = t2.copy()
    elif energies[i + 1] < energies[i] < energies[i - 1]:
        tangent = t1.copy()
    else:
        deltavmax = max(abs(energies[i + 1] - energies[i]),
                        abs(energies[i - 1] - energies[i]))
        deltavmin = min(abs(energies[i + 1] - energies[i]),
                        abs(energies[i - 1] - energies[i]))
        if energies[i + 1] > energies[i - 1]:
            tangent = t2 * deltavmax + t1 * deltavmin
        else:
            tangent = t2 * deltavmin + t1 * deltavmax
    t1 = t2
    nt1 = nt2
    # Normalize the tangent vector
    tangent /= np.linalg.norm(tangent)
    #print('tangent', i, tangent)
    tangent_spline = dd_ds(s[i])
    tangent_spline /= np.linalg.norm(tangent_spline)
    #print('tangent_spline', i, tangent_spline)

    print(np.linalg.norm(tangent_spline - tangent.reshape(-1), np.inf))

