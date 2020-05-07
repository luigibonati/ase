from ase.build import fcc100, add_adsorbate
from ase.constraints import FixAtoms, FixedPlane
from ase.calculators.emt import EMT
from ase.optimize import QuasiNewton
import re
import numpy as np
from ase.cli.template import prec_round, sort2rank, slice_split, \
                             MapFormatter, num2sym, sym2num, \
                             Table, TableFormat

"""
An enumeration of test cases:

# of files = 1, 2
calculation outputs = 0, 1, 2
multiple images = 0, 1, 2

Under the constraint that # of files is greater than or equal to number
of calculation outputs and multiple images. Also, if there are two
files, one only cares about the case that both have the same number of
images and both have calculator outputs or not.

(1,0,1)
(1,1,1)
(2,0,0)
(2,0,2)
(2,2,0)
(2,2,2)

Tests for these cases and all command line options are done.
"""


def test_diff(cli):
    slab = fcc100('Al', size=(2, 2, 3))
    add_adsorbate(slab, 'Au', 1.7, 'hollow')
    slab.center(axis=2, vacuum=4.0)
    mask = [atom.tag > 1 for atom in slab]
    fixlayers = FixAtoms(mask=mask)
    plane = FixedPlane(-1, (1, 0, 0))
    slab.set_constraint([fixlayers, plane])
    slab.set_calculator(EMT())
    qn = QuasiNewton(slab, trajectory='AlAu.traj')
    qn.run(fmax=0.02)

    stdout = cli.ase('diff --as-csv AlAu.traj')

    r = c = -1
    for rowcount, row in enumerate(stdout.split('\n')):
        for colcount, col in enumerate(row.split(',')):
            if col == 'Δx':
                r = rowcount + 2
                c = colcount
            if (rowcount == r) & (colcount == c):
                val = col
                break
    assert(float(val) == 0.)

    cli.ase('diff AlAu.traj -c')
    cli.ase('diff AlAu.traj@:1 AlAu.traj@1:2')
    cli.ase('diff AlAu.traj@:1 AlAu.traj@1:2 -c')
    cli.ase('diff AlAu.traj@:2 AlAu.traj@2:4')
    stdout = cli.ase('diff AlAu.traj@:2 AlAu.traj@2:4 -c --rank-order dfx --as-csv')
    stdout = [row.split(',') for row in stdout.split('\n')]
    stdout = [row for row in stdout if len(row) > 4]

    header = stdout[0]
    body = stdout[1:len(slab)] # note tables are appended
    for c in range(len(header)):
        if header[c] == 'Δfx':
            break
    col = [float(row[c]) for row in body]
    assert( (col[:-1] <= col[1:]) )

    # template command line options
    stdout = cli.ase('diff AlAu.traj@:1 AlAu.traj@:2 -c '
            '--template p1x,p2x,dx,f1x,f2x,dfx')
    stdout = stdout.split('\n')

    for counter, row in enumerate(stdout):
        if '=' in row: # default toprule
            header = stdout[counter + 1]
            break
    header = re.sub(r'\s+',',',header).split(',')[1:-1]
    assert(header == ['p1x','p2x','Δx','f1x','f2x','Δfx']) 

    cli.ase('diff AlAu.traj -c --template p1x,f1x,p1y,f1y:0:-1,p1z,f1z,p1,f1 '
            '--max-lines 6 --summary-functions rmsd')

def test_template():
    """Test functions used in the template"""
    num = 1.55749
    rnum = [prec_round(num,i) for i in range(1,6)]
    assert(rnum == [1.6, 1.56, 1.557, 1.5575, 1.55749])
    blarray = [4,3,1,0,2] == sort2rank([3,2,4,1,0]) # sort2rank outputs numpy array
    assert(blarray.all())
    assert(slice_split('a@1:3:1') == ('a',slice(1,3,1)))
    
    sym = 'H'
    num = sym2num[sym]
    mf = MapFormatter().format
    sym2 = mf('{:h}', num)
    assert(sym == mf('{:h}',num))
