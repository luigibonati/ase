from ase.calculators.emt import EMT
from ase.constraints import FixInternals
from ase.optimize.bfgs import BFGS
from ase.build import molecule
import copy
import pytest


def get_bondcombo(atoms, bondcombo_def):
    return sum([defin[2] * atoms.get_distance(defin[0], defin[1]) for
                defin in bondcombo_def])


def setup_test():
    atoms = molecule('CH3CH2OH', vacuum=5.0)
    atoms.rattle(stdev=0.3)

    # Angles, Bonds, Dihedrals are built up with pairs of constraint
    # value and indices defining the constraint
    # Linear combinations of bond lengths are built up similarly with the
    # coefficients appended to the indices defining the constraint

    # Fix bond between atoms 1 and 2 to 1.4
    bond_def = [1, 2]
    target_bond = 1.4

    # Fix angle to whatever it was from the start
    angle_def = [6, 0, 1]
    target_angle = atoms.get_angle(*angle_def)

    # Fix this dihedral angle to whatever it was from the start
    dihedral_def = [6, 0, 1, 2]
    target_dihedral = atoms.get_dihedral(*dihedral_def)

    # Fix linear combination of two bond lengths with atom indices 0-8 and
    # 0-6 with weighting coefficients 1.0 and -1.0 to the current value.
    # In other words, fulfil the following constraint:
    # 1 * atoms.get_distance(0, 8) -1 * atoms.get_distance(0, 6) = const.
    bondcombo_def = [[0, 8, 1.0], [0, 6, -1.0]]
    target_bondcombo = get_bondcombo(atoms, bondcombo_def)

    # Initialize constraint
    constr = FixInternals(bonds=[(target_bond, bond_def)],
                              angles_deg=[(target_angle, angle_def)],
                              dihedrals_deg=[(target_dihedral, dihedral_def)],
                              bondcombos=[(target_bondcombo, bondcombo_def)],
                              epsilon=1e-10)
    print(constr)
    return (atoms, constr, bond_def, target_bond, angle_def, target_angle,
            dihedral_def, target_dihedral, bondcombo_def, target_bondcombo)


def test_fixinternals():
    (atoms, constr, bond_def, target_bond, angle_def, target_angle,
     dihedral_def, target_dihedral, bondcombo_def,
     target_bondcombo) = setup_test()

    calc = EMT()

    opt = BFGS(atoms, trajectory='opt.traj', logfile='opt.log')

    previous_angle = atoms.get_angle(*angle_def)
    previous_dihedral = atoms.get_dihedral(*dihedral_def)
    previous_bondcombo = get_bondcombo(atoms, bondcombo_def)

    print('angle before', previous_angle)
    print('dihedral before', previous_dihedral)
    print('bond length before', atoms.get_distance(*bond_def))
    print('(target bondlength %s)', target_bond)
    print('linear bondcombination before', previous_bondcombo)

    atoms.calc = calc
    atoms.set_constraint(constr)
    print('-----Optimization-----')
    opt.run(fmax=0.01)

    new_angle = atoms.get_angle(*angle_def)
    new_dihedral = atoms.get_dihedral(*dihedral_def)
    new_bondlength = atoms.get_distance(*bond_def)
    new_bondcombo = get_bondcombo(atoms, bondcombo_def)

    print('angle after', new_angle)
    print('dihedral after', new_dihedral)
    print('bondlength after', new_bondlength)
    print('linear bondcombination after', new_bondcombo)

    err1 = new_angle - previous_angle
    err2 = new_dihedral - previous_dihedral
    err3 = new_bondlength - target_bond
    err4 = new_bondcombo - previous_bondcombo

    print('error in angle', repr(err1))
    print('error in dihedral', repr(err2))
    print('error in bondlength', repr(err3))
    print('error in bondcombo', repr(err4))

    assert err1 < 1e-11
    assert err2 < 1e-12
    assert err3 < 1e-12
    assert err4 < 1e-12


def test_index_shuffle():
    (atoms, constr, bond_def, target_bond, angle_def, target_angle,
     dihedral_def, target_dihedral, bondcombo_def,
     target_bondcombo) = setup_test()

    constr2 = copy.deepcopy(constr)

    # test no change, test constr.get_indices()
    assert all(a == b for a, b in zip(constr.get_indices(), (0, 1, 2, 6, 8)))
    constr.index_shuffle(atoms, range(len(atoms)))
    assert all(a == b for a, b in zip(constr.get_indices(), (0, 1, 2, 6, 8)))

    # test only bondcombo remains
    constr.index_shuffle(atoms, [0, 6, 8])
    assert constr.bondcombos[0][1] == [[0, 2, 1.0], [0, 1, -1.0]]
    
    # test full constraint is not part of new slice
    with pytest.raises(IndexError):
        constr.index_shuffle(atoms, [0])

    # test only bondcombo is not part of the new slice
    constr2.index_shuffle(atoms, [1, 2, 0, 6])
    assert constr2.bonds[0][1] == [0, 1]
    assert constr2.angles[0][1] == [3, 2, 0]
    assert constr2.dihedrals[0][1] == [3, 2, 0, 1]
