#import matplotlib.pyplot as plt

import pytest

from ase.calculators.emt import EMT
from ase.optimize import BFGSLineSearch, BFGS
from ase.build import bulk
from ase.neb import NEB, NEBTools

Ef_ref = 0.7621119906629144

@pytest.mark.parametrize('method', ['aseneb', 'improvedtangent', 'spline'])
def test_vacancy(method):
    N = 2
    initial = bulk('Cu', cubic=True)
    initial *= (N, N, N)
    pos = initial.get_positions()

    del initial[0]
    final = initial.copy()
    final.positions[0] = pos[0]

    initial.calc = EMT()
    final.calc = EMT()

    # Relax initial and final states
    qn = BFGSLineSearch(initial)
    qn.run(fmax=0.01)
    qn = BFGSLineSearch(final)
    qn.run(fmax=0.01)

    images = [initial]
    for image in range(3):
        image = initial.copy()
        image.calc = EMT()
        images.append(image)
    images.append(final)

    neb = NEB(images, method=method)
    neb.interpolate()
    qn = BFGS(neb)
    qn.run(fmax=0.05)

    nebtools = NEBTools(images)
    Ef_neb, dE_neb = nebtools.get_barrier(fit=False)
    print(Ef_neb, dE_neb)

    #fig = nebtools.plot_band()
    #plt.show()

    assert(abs(Ef_neb - Ef_ref) < 1e-3)
    assert(abs(dE_neb) < 1e-6)

    print(images[2].positions)

