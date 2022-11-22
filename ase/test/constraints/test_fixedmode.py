def test_fixedmode():
    """Test fixed mode can be set, turned into a dict, and
    back to a mode."""
    import numpy as np
    from ase.build import molecule
    from ase.calculators.emt import EMT
    from ase.optimize import BFGS
    from ase.vibrations import Vibrations
    from ase.constraints import FixedMode, dict2constraint

    # Create a vibrational mode.
    atoms = molecule('N2')
    atoms.calc = EMT()
    opt = BFGS(atoms)
    opt.run(fmax=0.01)
    vib = Vibrations(atoms)
    vib.run()
    mode = vib.get_mode(-1)

    # Test the constraint.
    constraint = FixedMode(mode)
    dict_constraint = constraint.todict()
    new_constraint = dict2constraint(dict_constraint)
    assert np.isclose(new_constraint.mode, constraint.mode).all()
