from ase import Atoms
from ase.autoneb import AutoNEB
from ase.build import fcc211, add_adsorbate
from ase.constraints import FixAtoms
from ase.neb import NEBTools
from ase.optimize import QuasiNewton
from ase.calculators.emt import EMT


def test_autoneb(asap3, testdir):
    EMT = asap3.EMT
    fmax = 0.02

    # Pt atom adsorbed in a hollow site:
    slab = fcc211('Pt', size=(3, 2, 2), vacuum=4.0)
    add_adsorbate(slab, 'Pt', 0.5, (-0.1, 2.7))

    # Fix second and third layers:
    slab.set_constraint(FixAtoms(range(6, 12)))

    # Use EMT potential:
    slab.calc = EMT()

    # Initial state:
    with QuasiNewton(slab, trajectory='neb000.traj') as qn:
        qn.run(fmax=fmax)

    # Final state:
    slab[-1].x += slab.get_cell()[0, 0]
    slab[-1].y += 2.8
    with QuasiNewton(slab, trajectory='neb001.traj') as qn:
        qn.run(fmax=fmax)

    # Stops PermissionError on Win32 for access to
    # the traj file that remains open.
    del qn

    def attach_calculators(images):
        for i in range(len(images)):
            images[i].calc = EMT()

    autoneb = AutoNEB(attach_calculators,
                      prefix='neb',
                      optimizer='BFGS',
                      n_simul=3,
                      n_max=7,
                      fmax=fmax,
                      k=0.5,
                      parallel=False,
                      maxsteps=[50, 1000])
    autoneb.run()

    nebtools = NEBTools(autoneb.all_images)
    assert abs(nebtools.get_barrier()[0] - 0.937) < 1e-3


def test_Au2Ag(testdir):
    def attach_calculators(images):
        for i in range(len(images)):
            images[i].calc = EMT()

    d = 4
    initial = Atoms('Au2Ag', positions=((-d, 0, 0), (0, 0, 0), (d, 0, 1)))
    middle = initial.copy()
    final = initial.copy()
    final[1].position[0] += d / 2

    attach_calculators([initial, middle, final])

    prefix = 'neb'
    
    fmax = 0.05
    for i, image in enumerate([initial, middle, final]):
        image.set_constraint(FixAtoms([0, 2]))
        opt = QuasiNewton(image)
        opt.run(fmax=fmax)
        image.write(f'neb00{i}.traj')
    
    autoneb = AutoNEB(attach_calculators,
                      prefix=prefix,
                      optimizer='BFGS',
                      n_simul=1,
                      n_max=7,
                      fmax=fmax,
                      k=0.5,
                      parallel=False,
                      maxsteps=[20, 1000])
    autoneb.run()
    
    from ase import io
    with io.Trajectory('all_images.traj', 'w') as traj:
        for im in autoneb.all_images:
            traj.write(im)
            
