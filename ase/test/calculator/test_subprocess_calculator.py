from ase.calculators.subprocesscalculator import wrap_subprocess
from ase.calculators.emt import EMT
from ase.build import bulk
from ase.optimize import BFGS
from ase.constraints import ExpCellFilter

def test_subprocess_calculator():
    emt = EMT()
    atoms = bulk('Au') * (2, 2, 2)
    atoms.rattle(stdev=0.05, seed=2)


    def get_fmax(forces):
        return max((forces**2).sum(axis=1)**0.5)

    #opt = BFGS(ExpCellFilter(atoms), trajectory='opt.traj')
    opt = BFGS(atoms, trajectory='opt.traj')
    with wrap_subprocess(emt) as atoms.calc:
        fmax0 = get_fmax(atoms.get_forces())
        print('start force', fmax0)
        assert fmax0 > 0.05
        opt.run(fmax=0.05)

    fmax_now = get_fmax(atoms.get_forces())
    print('end force', fmax_now)
    assert fmax_now < 0.05
