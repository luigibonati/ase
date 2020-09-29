from ase.optimize.activelearning.gpfp.fingerprint import RadialAngularFP
from ase.build import fcc111
import numpy as np

if __name__ == '__main__':

    """
    Test that fingerprint behaves as wanted with
    periodic boundary conditions, i.e. that re-centering
    atoms does not affect the fingerprint nor its gradients.
    """

    import time
    t0 = time.time()

    # Create slab:
    slab = fcc111('Ag', size=(2, 1, 2))
    slab[-4].symbol = 'Au'
    slab[-2].symbol = 'Au'
    slab.center(axis=2, vacuum=4.0)
    # slab.rattle(0.05)
    slab.pbc = (True, True, False)
    print("Number of atoms: ", len(slab))

    fp = RadialAngularFP
    params = dict(limit=4.0, Rlimit=3.6, delta=0.5, ascale=0.2)

    fp0 = fp(**params)
    fp0.set_atoms(slab)
    vec0 = fp0.vector

    slab.positions += np.diag(slab.get_cell()) * 0.1
    fp1 = fp(**params)
    fp1.set_atoms(slab)
    vec1 = fp1.vector

    d = fp0.distance(fp0, fp1)

    from ase.visualize import view
    view(slab)

    # from matplotlib import pyplot as plt
    # plt.plot(vec0)
    # plt.plot(vec1)
    # plt.show()


    assert(d < 1e-8)

    assert(np.allclose(vec0, vec1, atol=1e-8))

    assert(np.allclose(fp0.gradients, fp1.gradients, atol=1e-8))

    assert(np.allclose(fp0.anglegradients, fp1.anglegradients, atol=1e-8))

    t1 = time.time()
    print("time: {:.06f} sec".format(t1 - t0))
