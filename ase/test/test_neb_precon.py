import functools
import pytest

from ase.calculators.lj import LennardJones
from ase.optimize import BFGSLineSearch, BFGS, FIRE
from ase.build import bulk
from ase.neb import NEB, NEBTools
from ase.geometry.geometry import find_mic
from ase.constraints import FixBondLength

bulk_at = bulk("Cu", cubic=True)
sigma = (bulk_at*2).get_distance(0, 1)*(2.**(-1./6))
calc = lambda: LennardJones(sigma=sigma, epsilon=0.05)

def setup_images(N_intermediate=3):
    initial = bulk('Cu', cubic=True)
    initial *= 2
    pos = initial.get_positions()

    del initial[0]
    final = initial.copy()
    final.positions[0] = pos[0]

    initial.calc = calc()
    final.calc = calc()

    # Relax initial and final states
    qn = BFGSLineSearch(initial)
    qn.run(fmax=0.01)
    qn = BFGSLineSearch(final)
    qn.run(fmax=0.01)

    images = [initial]
    for image in range(N_intermediate):
        image = initial.copy()
        image.calc = calc()
        images.append(image)
    images.append(final)

    neb = NEB(images)
    neb.interpolate()
    return neb.images

@functools.lru_cache(maxsize=None)
def ref_vacancy():
    # use distance from moving atom to one of its neighbours as reaction coord
    # relax intermediate image to the saddle point using a bondlength constraint
    initial, saddle, final = setup_images(N_intermediate=1)
    saddle.calc = calc()
    saddle.set_constraint(FixBondLength(0, 11))
    opt = FIRE(saddle)
    opt.run(fmax=1e-3)
    nebtools = NEBTools([initial, saddle, final])
    Ef_ref, dE_ref = nebtools.get_barrier(fit=False)
    print('REF:', Ef_ref, dE_ref)
    return Ef_ref, dE_ref, saddle

@pytest.mark.slow()
@pytest.mark.parametrize('method, N_intermediate, precon',
                         [('aseneb', 3, None),
                          ('aseneb', 5, None),
                          ('improvedtangent', 3, None),
                          ('improvedtangent', 5, None),
                          ('spline', 3, None),
                          ('spline', 5, None),
                          ('spline', 3, 'Exp')])
def test_vacancy(method, N_intermediate, precon):
    Ef_ref, dE_ref, saddle_ref = ref_vacancy()

    # now relax the NEB band for comparison
    images = setup_images()
    neb = NEB(images)
    qn = BFGS(neb)
    qn.run(fmax=1e-3)

    nebtools = NEBTools(images)
    Ef_neb, dE_neb = nebtools.get_barrier(fit=False)
    print('NEB:', Ef_neb, dE_neb)

    assert abs(Ef_neb - Ef_ref) < 1e-3
    assert abs(dE_neb - dE_ref) < 1e-3

    # true saddle point known by symmetry
    vdiff, _ = find_mic(images[2].positions - saddle_ref.positions,
                        images[2].cell)ey6
    assert abs(vdiff).max() < 1e-2

