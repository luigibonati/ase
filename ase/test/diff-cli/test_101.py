#2 non-calculator files
def test_101(cli):
    from ase.build import fcc100
    from ase.io import write

    slab = fcc100('Al', size=(2, 2, 3))
    slab2 = slab.copy()
    slab2.positions += [0,0,1]
    write('slab.cif',[slab,slab2])

    stdout = cli.ase('diff slab.cif')

    stdoutref="""images 1-0
===============================================================
  index   element    Δx       Δy       Δz        Δ     rank Δ  
---------------------------------------------------------------
    0       Al     0.0E+00  0.0E+00  1.0E+00  1.0E+00     0    
    1       Al     0.0E+00  0.0E+00  1.0E+00  1.0E+00     1    
    2       Al     0.0E+00  0.0E+00  1.0E+00  1.0E+00     2    
    3       Al     0.0E+00  0.0E+00  1.0E+00  1.0E+00     3    
    4       Al     0.0E+00  0.0E+00  1.0E+00  1.0E+00     4    
    5       Al     0.0E+00  0.0E+00  1.0E+00  1.0E+00     5    
    6       Al     0.0E+00  0.0E+00  1.0E+00  1.0E+00     6    
    7       Al     0.0E+00  0.0E+00  1.0E+00  1.0E+00     7    
    8       Al     0.0E+00  0.0E+00  1.0E+00  1.0E+00     8    
    9       Al     0.0E+00  0.0E+00  1.0E+00  1.0E+00     9    
   10       Al     0.0E+00  0.0E+00  1.0E+00  1.0E+00    10    
   11       Al     0.0E+00  0.0E+00  1.0E+00  1.0E+00    11    
===============================================================
RMSD=+1.0E+00"""

    assert(stdout.strip() == stdoutref.strip())
