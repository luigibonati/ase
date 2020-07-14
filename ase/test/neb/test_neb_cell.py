def test_neb_cell():
    from ase import Atoms
    from ase.neb import NEB
    import numpy as np

    initial = Atoms('H', positions=[(1,0.1,0.1)],cell=[[1,0,0],[0,1,0],[0,0,1]],pbc=True)
    final = Atoms('H', positions=[(2,0.2,0.1)],cell=[[2,0,0],[0,2,0],[0,0,2]],pbc=True)
    images = [initial.copy()]
    images += [initial.copy()]
    images += [final.copy()]
    neb = NEB(images)
    neb.interpolate()
    assert np.allclose(images[1].positions, [1.5,0.15,0.1])
    assert np.allclose(images[1].cell, initial.cell)
    images = [initial.copy()]
    images += [initial.copy()]
    images += [final.copy()]
    neb = NEB(images)
    neb.interpolate(use_scaled_coord=True)
    assert np.allclose(images[1].positions, [1.0,0.1,0.075])
    assert np.allclose(images[1].cell, initial.cell)
    images = [initial.copy()]
    images += [initial.copy()]
    images += [final.copy()]
    neb = NEB(images)
    neb.interpolate(interpolate_cell=True)
    assert np.allclose(images[1].positions, [1.5,0.15,0.1])
    assert np.allclose(images[1].cell, initial.cell*1.5)
    images = [initial.copy()]
    images += [initial.copy()]
    images += [final.copy()]
    neb = NEB(images)
    neb.interpolate(interpolate_cell=True,use_scaled_coord=True)
    assert np.allclose(images[1].positions, [1.5,0.15,0.1125])
    assert np.allclose(images[1].cell, initial.cell*1.5)
