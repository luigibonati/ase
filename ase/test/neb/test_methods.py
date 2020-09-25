import numpy as np
import pytest

from ase.calculators.morse import MorsePotential
from ase.calculators.loggingcalc import LoggingCalculator
from ase.optimize import BFGS, FIRE, ODE12r
from ase.build import bulk
from ase.neb import NEB, NEBTools, NEBOptimizer
from ase.geometry.geometry import find_mic
from ase.constraints import FixBondLength
from ase.geometry.geometry import get_distances

import json
from ase.utils.forcecurve import fit_images


def calc():
    return LoggingCalculator(MorsePotential(A=4.0, epsilon=1.0, r0=2.55))


def setup_images(N_intermediate=3):
    N_cell = 2
    initial = bulk('Cu', cubic=True)
    initial *= N_cell

    # place vacancy near centre of cell
    D, D_len = get_distances(np.diag(initial.cell)/2,
                             [initial.positions[i] for i in range(len(initial))],
                             initial.cell, initial.pbc)
    vac_index = D_len.argmin()
    vac_pos = initial.positions[vac_index]
    del initial[vac_index]

    # identify two opposing nearest neighbours of the vacancy
    D, D_len = get_distances(vac_pos,
                             [initial.positions[i] for i in range(len(initial))],
                             initial.cell, initial.pbc)
    D = D[0, :]
    D_len = D_len[0, :]

    nn_mask = np.abs(D_len - D_len.min()) < 1e-8
    i1 = nn_mask.nonzero()[0][0]
    i2 = ((D + D[i1])**2).sum(axis=1).argmin()

    print(f'vac_index={vac_index} i1={i1} i2={i2} '
          f'distance={initial.get_distance(i1, i2, mic=True)}')

    final = initial.copy()
    final.positions[i1] = vac_pos

    initial.calc = calc()
    final.calc = calc()

    qn = ODE12r(initial)
    qn.run(fmax=1e-3)
    qn = ODE12r(final)
    qn.run(fmax=1e-3)

    images = [initial]
    for image in range(N_intermediate):
        image = initial.copy()
        image.calc = calc()
        images.append(image)
        # TODO: rattle images to make test harder!
        # image.rattle()
    images.append(final)

    neb = NEB(images)
    neb.interpolate()

    return neb.images, i1, i2

@pytest.fixture
def ref_vacancy():
    # use distance from moving atom to one of its neighbours as reaction coord
    # relax intermediate image to the saddle point using a bondlength constraint
    (initial, saddle, final), i1, i2 = setup_images(N_intermediate=1)
    saddle.calc = calc()
    saddle.set_constraint(FixBondLength(i1, i2))
    opt = ODE12r(saddle)
    opt.run(fmax=1e-2)
    nebtools = NEBTools([initial, saddle, final])
    Ef_ref, dE_ref = nebtools.get_barrier(fit=False)
    print('REF:', Ef_ref, dE_ref)
    return Ef_ref, dE_ref, saddle

@pytest.mark.slow()
@pytest.mark.filterwarnings('ignore:estimate_mu')
@pytest.mark.parametrize('method, optimizer, precon, N_intermediate, optmethod',
                         [
                          ('aseneb', BFGS, None, 3, None),
                          ('improvedtangent', BFGS, None, 3, None),
                          ('aseneb', FIRE, None, 3, None),
                          ('improvedtangent', FIRE, None, 3, None),
                          ('spline', NEBOptimizer, None, 3, 'ODE'),
                          ('string', NEBOptimizer, None, 3, 'ODE'),
                          ('spline', NEBOptimizer, 'Exp', 3, 'ODE'),
                          ('string', NEBOptimizer, 'Exp', 3, 'ODE'),
                          ('spline', NEBOptimizer, None, 3, 'krylov'),
                          ('spline', NEBOptimizer, 'Exp', 3, 'krylov'),
                         ])
def test_neb_methods(method, optimizer, precon, N_intermediate, optmethod, ref_vacancy):
    # unpack the reference result
    Ef_ref, dE_ref, saddle_ref = ref_vacancy

    # now relax the MEP for comparison
    images, _, _ = setup_images(N_intermediate)

    k = 0.1
    if precon == 'Exp':
        k = 0.01
    mep = NEB(images, k=k, method=method, precon=precon)
    if optmethod is not None:
        opt = optimizer(mep, method=optmethod)
    else:
        opt = optimizer(mep)
    opt.fmax_history = []
    opt.run(fmax=1e-3, steps=500)

    nebtools = NEBTools(images)
    Ef, dE = nebtools.get_barrier(fit=False)
    print(f'{method},{optimizer.__name__},{precon} '
          f'=> Ef = {Ef:.3f}, dE = {dE:.3f}')

    forcefit = fit_images(images)

    # retain biggest force across all images
    fmax_history = list(np.max([mep.images[i].calc.fmax['(none)'] 
                                for i in range(1, mep.nimages-1)], axis=0))

    # add up walltime across images
    walltime = list(np.sum([mep.images[i].calc.fmax['(none)'] 
                                for i in range(1, mep.nimages-1)], axis=0))

    # output_dir = '/Users/jameskermode/gits/ase/ase/test'  # FIXME avoid this hack
    # with open(f'{output_dir}/MEP_{method}_{optimizer.__name__}_{optmethod}_{precon}_{N_intermediate}.json', 'w') as f:
    #     json.dump({'fmax_history': fmax_history,
    #                'walltime': walltime,
    #                'method': method,
    #                'optmethod': optmethod,
    #                'precon': precon,
    #                'optimizer': optimizer.__name__,
    #                'N_intermediate': N_intermediate,
    #                'path': forcefit.path,
    #                'energies': forcefit.energies.tolist(),
    #                'fit_path': forcefit.fit_path.tolist(),
    #                'fit_energies': forcefit.fit_energies.tolist(),
    #                'lines': np.array(forcefit.lines).tolist(),
    #                'Ef': Ef,
    #                'dE': dE}, f)

    assert abs(Ef - Ef_ref) < 1e-2
    assert abs(dE - dE_ref) < 1e-2

    centre = 1 + N_intermediate // 2
    vdiff, _ = find_mic(images[centre].positions - saddle_ref.positions,
                        images[centre].cell)
    assert abs(vdiff).max() < 1e-2


def test_precon_initialisation():
    images, _, _ = setup_images(5)
    mep = NEB(images, method='spline', precon='Exp')
    assert len(mep.precon) == len(mep.images)
    assert len(set(mep.precon)) == len(mep.precon)
    assert mep.precon[0].mu == mep.precon[1].mu
