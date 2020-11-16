def test_gen():
    import numpy as np

    from ase import Atoms
    from ase.io import read, write

    a = Atoms(symbols='OCO', pbc=False,
              positions=[[-0.10058585, 1.17858096, 0.32044094],
                         [-0.09987592, 0.00262508, 0.27479323],
                         [0.36599183, -0.88037721, -0.14084089]])

    # First, a non-periodic structure. Also with cell vectors,
    # reading the .gen file should yield a non-periodic structure
    for cell in (None, [[2.5, 0., 0.], [2., 4., 0.], [1., 2., 3.]]):
        a.set_cell(cell)
        write('test.gen', a)

        b = read('test.gen')
        assert np.all(b.numbers == a.numbers)
        assert np.allclose(b.positions, a.positions)
        assert np.all(b.pbc == a.pbc)
        assert np.allclose(b.cell, 0.)

    # Mixed and full periodicity. This should in both cases
    # yield a periodic structure when reading the .gen file
    for pbc in ([True, True, False], True):
        a.set_pbc(pbc)
        write('test.gen', a)

        b = read('test.gen')
        assert np.all(b.numbers == a.numbers)
        assert np.allclose(b.positions, a.positions)
        assert np.all(b.pbc)
        assert np.allclose(b.cell, a.cell)

    # Try with multiple images. This is not supported by the
    # format and should fail
    try:
        write('test.gen', [a, a])
    except ValueError:
        pass
    else:
        assert False
