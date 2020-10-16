def test_dihedralconstraint():
    from ase.calculators.emt import EMT
    from ase.constraints import FixInternals
    from ase.optimize.bfgs import BFGS
    from ase.build import molecule

    system = molecule('CH3CH2OH', vacuum=5.0)
    system.rattle(stdev=0.3)

    # Angles, Bonds, Dihedrals are built up with pairs of constraint
    # value and indices defining the constraint
    # Linear combinations of bond lengths are built up similarly with the
    # coefficients appended to the indices defining the constraint

    # Fix this dihedral angle to whatever it was from the start
    indices = [6, 0, 1, 2]
    dihedral1 = system.get_dihedral(*indices)

    # Fix angle to whatever it was from the start
    indices2 = [6, 0, 1]
    angle1 = system.get_angle(*indices2)

    # Fix bond between atoms 1 and 2 to 1.4
    target_bondlength = 1.4
    indices_bondlength = [1, 2]

    # Fix linear combination of two bond lengths with atom indices 0-8 and
    # 0-6 with weighting coefficients 1.0 and -1.0 to the current value.
    # In other words, fulfil the following constraint:
    # 1 * system.get_distance(0, 8) -1 * system.get_distance(0, 6) = const.
    bondcombo_def = [[0, 8, 1.0], [0, 6, -1.0]]

    def get_bondcombo(system, bondcombo_def):
        return sum([defin[2] * system.get_distance(defin[0], defin[1]) for
                    defin in bondcombo_def])

    target_bondcombo = get_bondcombo(system, bondcombo_def)

    constraint = FixInternals(bonds=[(target_bondlength, indices_bondlength)],
                              angles_deg=[(angle1, indices2)],
                              dihedrals_deg=[(dihedral1, indices)],
                              bondcombos=[(target_bondcombo, bondcombo_def)],
                              epsilon=1e-10)

    print(constraint)

    calc = EMT()

    opt = BFGS(system, trajectory='opt.traj', logfile='opt.log')

    previous_angle = system.get_angle(*indices2)
    previous_dihedral = system.get_dihedral(*indices)
    previous_bondcombo = get_bondcombo(system, bondcombo_def)

    print('angle before', previous_angle)
    print('dihedral before', previous_dihedral)
    print('bond length before', system.get_distance(*indices_bondlength))
    print('(target bondlength %s)', target_bondlength)
    print('linear bondcombination before', previous_bondcombo)

    system.calc = calc
    system.set_constraint(constraint)
    print('-----Optimization-----')
    opt.run(fmax=0.01)

    new_angle = system.get_angle(*indices2)
    new_dihedral = system.get_dihedral(*indices)
    new_bondlength = system.get_distance(*indices_bondlength)
    new_bondcombo = get_bondcombo(system, bondcombo_def)

    print('angle after', new_angle)
    print('dihedral after', new_dihedral)
    print('bondlength after', new_bondlength)
    print('linear bondcombination after', new_bondcombo)

    err1 = new_angle - previous_angle
    err2 = new_dihedral - previous_dihedral
    err3 = new_bondlength - target_bondlength
    err4 = new_bondcombo - previous_bondcombo

    print('error in angle', repr(err1))
    print('error in dihedral', repr(err2))
    print('error in bondlength', repr(err3))
    print('error in bondcombo', repr(err4))

    assert err1 < 1e-11
    assert err2 < 1e-12
    assert err3 < 1e-12
    assert err4 < 1e-12
