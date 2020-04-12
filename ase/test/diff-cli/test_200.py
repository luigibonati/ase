# 2 non-calculator files


def test_200(cli):
    from ase.build import fcc100
    from ase.io import write

    slab = fcc100('Al', size=(2, 2, 3))
    write('slab200-1.cif', slab)
    import numpy.random as random
    random.seed(25)
    slab.positions += random.random((len(slab), 3))
    write('slab200-2.cif', slab)

    stdout = cli.ase('diff slab200-1.cif slab200-2.cif')
