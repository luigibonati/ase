# creates: spacegroup-al.png spacegroup-fe.png spacegroup-rutile.png spacegroup-cosb3.png spacegroup-mg.png spacegroup-skutterudite.png spacegroup-diamond.png spacegroup-nacl.png
import runpy

import ase.io

for name in ['al', 'mg', 'fe', 'diamond', 'nacl', 'rutile', 'skutterudite']:
    py = 'spacegroup-{0}.py'.format(name)
    dct = runpy.run_path(py)
    atoms = dct[name]
    ase.io.write('spacegroup-%s.pov' % name,
                 atoms,
                 transparent=False,
                 run_povray=True,
                 # canvas_width=128,
                 rotation='10x,-10y',
                 # celllinewidth=0.02,
                 celllinewidth=0.05)

runpy.run_path('spacegroup-cosb3.py')
