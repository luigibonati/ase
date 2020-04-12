# 2 non-calculator files
def test_101(cli):#,tmpdir):
    from ase.build import fcc100
    from ase.io import write

    slab = fcc100('Al', size=(2, 2, 3))
    slab2 = slab.copy()
    slab2.positions += [0, 0, 1]
    write('101.cif', [slab, slab2])

    stdout = cli.ase('diff --as-csv 101.cif')
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
