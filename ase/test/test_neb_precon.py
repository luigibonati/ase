import pytest
import numpy
from ase.calculators.emt import EMT
from ase.optimize import Quasinewton
from ase.build import fcc211, add_adsorbate 
from ase.neb import NEB



def test_precon_neb():
    N = 3
    Initial = bulk('Cu', cubic=True)
    Initial *= (N, N, N)

    Final = Initial.copy() 
    
    Initial[0].del()
    Final[1].del()


    #Attach a writer
    def initial(atoms=Initial):
        ase.io.write("initial.xyz", atoms, append=True)
    def final(atoms=Final):
        ase.io.write("final.xyz", atoms, append=True)

    # Initial state:
    qn = QuasiNewton(Initial)
    qn.attach(initial)
    qn.run(fmax=0.05)

    # Final state:
    qn = QuasiNewton(Final)
    qn.attach(final)
    qn.run(fmax=0.05)


    images = [Initial]
    for image in range(3):
        image = Initial.copy()
        image.set_calculator(calc)
        images.append(image)

    Initial_precon = Initial.copy()
    Final_precon = Final.copy()
    
    neb = NEB(images, method = 'improvedtangent')
    neb_precon = NEB(images_precon, method = 'precon')
    
    #Run neb
    neb.interpolate()
    neb_precon.interpolate()
    qn = BFGS(neb)
    qn_precon = BFGS(neb_precon)

    qn.run(fmax = 0.05)
    qn_precon.run(fmax = 0.05)


    images = neb.images
    images_precon = neb.images_precon

    nebtools = NEBTools(images)
    Ef_neb, dE_neb = nebtools.get_barrier(fit=False)

    nebtools_preon = NEBTools(images_precon)
    Ef_neb_precon, dE_neb_precon = nebtools_precon.get_barrier(fit=False)


    assert(abs(Ef_neb_precon - Ef_neb) < 0.05)
    assert(abs(dE_neb_precon - dE_neb) < 0.05)

    for i in len(images[2]):
        assert(abs(images[2].get_positions()[i] - images_precon[2].get_positions()[i]) < 0.05)
