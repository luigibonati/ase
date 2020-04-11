#2 non-calculator files
def test_101(cli)
    from ase.build import fcc100
    from ase.io import write

    slab = fcc100('Al', size=(2, 2, 3))
    slab2 = slab.copy()
    slab2.positions += [0,0,1]
    write('slab.cif',[slab,slab2])

    stdout = cli.ase('ase diff slab.cif')
