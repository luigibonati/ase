# 2 non-calculator files


def test_202(cli):
    from ase.build import fcc100
    from ase.io import write

    slab = fcc100('Al', size=(2, 2, 3))
    write('slab202-1.cif', [slab, slab, slab])
    import numpy.random as random
    random.seed(543)
    slab.positions += random.random((len(slab), 3))
    write('slab202-2.cif', [slab, slab, slab])

    stdout = cli.ase('diff slab202-1.cif slab202-2.cif')
