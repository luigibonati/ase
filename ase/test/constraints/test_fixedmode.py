def test_fixedmode():
    """Test fixed mode can be set, turned into a dict, and
    back to a mode."""
    import numpy as np
    from ase.build import molecule
    from ase.constraints import FixedMode, dict2constraint

    # Create a simple mode.
    atoms = molecule('N2')
    atoms2 = atoms.copy()
    atoms2.positions += [(.3, .5, .1), (-.1, .2, -.1)]
    mode = atoms2.positions - atoms.positions

    # Test the constraint.
    constraint = FixedMode(mode)
    dict_constraint = constraint.todict()
    new_constraint = dict2constraint(dict_constraint)
    assert np.isclose(new_constraint.mode, constraint.mode).all()
