from ase import Atoms
from ase.constraints import FixInternals
from ase.calculators.emt import EMT
from ase.optimize.climbfixinternals import ClimbFixInternals

def setup_atoms():
    """Setup transition state search for an SN2 reaction:
    The substitution of MeCl by Cl (in this demonstration Cu is used instead of
    Cl, because the EMT calculator does not support Cl)."""
    positions = [[7.07784622, 6.36737904, 8.47511833],
                 [7.50617774, 5.97274016, 7.55045035],
                 [6.00151453, 6.51727226, 8.36050116],
                 [7.56351536, 7.30754202, 8.74801782],
                 [6.58494379, 8.43777561, 6.21601465],
                 [7.36058358, 5.14853782, 9.81663707]]
    atoms = Atoms(symbols='CH3Cu2', positions=positions)
    return atoms

def get_combo_value(atoms, combo):
    """Return current value of linear combination of bonds lengths, angles or
    dihedrals defined as bondcombo, anglecombo or dihedralcombo
    (see the FixInternals class)."""
    coord_type = len(combo)
    if coord_type == 2:
        get_value = atoms.get_distance
    elif coord_type == 3:
        get_value = atoms.get_angle
    elif coord_type == 4:
        get_value = atoms.get_dihedral
    return sum([defin[coord_type] * get_value(*defin[:coord_type]) for
                defin in combo])

def test_climb_fix_internals():
    """Climb along the constrained bondcombo coordinate while optimizing the
    remaining degrees of freedom after each climbing step.
    For the definition of constrained internal coordinates see the
    documentation of the classes FixInternals and ClimbFixInternals."""
    atoms = setup_atoms()
    atoms.calc = EMT()

    # Define reaction coordinate via linear combination of bond lengths
    reaction_coord = [[0, 4, -1.0], [0, 5, 1.0]]  # -1 * bond_1 + 1 * bond_2
    bondcombo = [get_combo_value(atoms, reaction_coord), reaction_coord]
    atoms.set_constraint(FixInternals(bondcombos=[bondcombo]))

    # Optimizer for transition state search along reaction coordinate
    dyn = ClimbFixInternals(atoms,
                            climb_coordinate=['FixBondCombo',
                                              [[0, 4], [0, 5]]])
    
    # Converge to a saddle point
    dyn.run(fmax=0.01)

    # Test the results
    assert abs(get_combo_value(atoms, reaction_coord)) < 0.003

def test_initialization_with_different_constraints():
    """Bond length, angle and dihedral coordinates have to be provided as
    nested lists to the optimizer."""
    atoms = setup_atoms()
    bond = FixInternals(bonds=[[1.2, [0, 1]]])
    angle = FixInternals(angles_deg=[[atoms.get_angle(0, 1, 2), [0, 1, 2]]])
    dihedral = FixInternals(dihedrals_deg=[[atoms.get_dihedral(0, 1, 2, 3), [0, 1, 2, 3]]])
    names = ['FixBondLengthAlt', 'FixAngle', 'FixDihedral']
    for i, constr in enumerate([bond, angle, dihedral]):
        atoms.set_constraint()
        atoms.set_constraint(constr)
        dyn = ClimbFixInternals(atoms,
                                climb_coordinate=[names[i],
                                                  [list(range(0, 2+i))]])
    bc = [[0, 4, -1.0], [0, 5, 1.0]]
    bondcombo = FixInternals(bondcombos=[get_combo_value(atoms, bc), bc])
    ac = [[5, 0, 3, -1.0], [5, 0, 2, 1.0]]
    anglecombo = FixInternals(anglecombos=[get_combo_value(atoms, ac), ac])
    dc = [[5, 0, 3, 4, -1.0], [5, 0, 2, 4, 1.0]]
    dihedralcombo = FixInternals(dihedralcombos=[get_combo_value(atoms, dc), dc])
    names = ['FixBondCombo', 'FixAngleCombo', 'FixDihedralCombo']
    coord = [bc, ac, dc]
    for i, constr in enumerate([bondcombo, anglecombo, dihedralcombo]):
        atoms.set_constraint()
        atoms.set_constraint(constr)
        print(i)
        print(names[i])
        print(coord[i])
        print([c[:-1] for c in coord[i]])
        dyn = ClimbFixInternals(atoms,
                                climb_coordinate=[names[i],
                                                 [c[:-1] for c in coord[i]]])
