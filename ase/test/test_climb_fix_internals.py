from ase.build import fcc100, add_adsorbate
from ase.constraints import FixAtoms, FixInternals
from ase.calculators.emt import EMT
from ase.optimize.climbfixinternals import ClimbFixInternals
from ase.vibrations import Vibrations
import numpy as np

# from ase.io import read
# from ase.visualize import view


def setup_atoms():
    """Setup transition state search for the diffusion barrier for a Pt atom
    on a Pt surface."""
    atoms = fcc100('Pt', size=(2, 2, 1), vacuum=10.0)
    add_adsorbate(atoms, 'Pt', 1.611, 'hollow')
    atoms.set_constraint(FixAtoms(list(range(4))))  # freeze the slab
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
    #reaction_coord = [[0, 4, -1.0], [0, 5, 1.0]]  # -1 * bond_1 + 1 * bond_2
    reaction_coord = [[0, 4, 1.0], [1, 4, 1.0]]  # 1 * bond_1 + 1 * bond_2
    bondcombo = [get_combo_value(atoms, reaction_coord), reaction_coord]
    #atoms.set_constraint(atoms.constraints + [FixInternals(bondcombos=[bondcombo])])
    atoms.set_constraint([FixInternals(bondcombos=[bondcombo])] + atoms.constraints)

    # Optimizer for transition state search along reaction coordinate
    dyn = ClimbFixInternals(atoms,
                            climb_coordinate=['FixBondCombo',
                                              #[[0, 4], [0, 5]]],
                                              [[0, 4], [1, 4]]],
                            optB_kwargs={'logfile': '-'}, optB_fmax=0.05,
                            trajectory='opt.traj')

    # Converge to a saddle point
    dyn.run(fmax=0.005)

    # Visualize transition state search
    # a = read('opt.traj', index=':')
    # view(a)

    # Validate transition state by one imaginary vibrational mode
    vib = Vibrations(atoms, indices=[4])
    vib.run()
    assert ((np.imag(vib.get_energies()) > 0) == [True, False, False]).all()

    # Visualize imaginary vibrational mode
    # vib.write_mode(0)
    # v = read('vib.0.traj', index=':')
    # view(v)


def test_initialization_with_different_constraints():
    """Remember to provide reaction coordinates as nested lists.
    Definitions in this example make no sense but initialization is checked."""
    atoms = setup_atoms()
    bond = FixInternals(bonds=[[atoms.get_distance(0, 1), [0, 1]]])
    angle = FixInternals(angles_deg=[[atoms.get_angle(0, 1, 2), [0, 1, 2]]])
    dihedral = FixInternals(dihedrals_deg=[[atoms.get_dihedral(0, 1, 2, 3),
                                            [0, 1, 2, 3]]])
    names = ['FixBondLengthAlt', 'FixAngle', 'FixDihedral']
    for i, constr in enumerate([bond, angle, dihedral]):
        atoms.set_constraint()
        atoms.set_constraint(constr)
        ClimbFixInternals(atoms,
                          climb_coordinate=[names[i],
                                            [list(range(0, 2 + i))]])
    bc = [[0, 4, -1.0], [1, 4, 1.0]]
    bondcombo = FixInternals(bondcombos=[[get_combo_value(atoms, bc), bc]])
    ac = [[0, 1, 2, -1.0], [0, 2, 3, 1.0]]
    anglecombo = FixInternals(anglecombos=[[get_combo_value(atoms, ac), ac]])
    dc = [[0, 1, 2, 3, -1.0], [0, 1, 4, 3, 1.0]]
    dihedralcombo = FixInternals(dihedralcombos=[[get_combo_value(atoms, dc),
                                                  dc]])
    names = ['FixBondCombo', 'FixAngleCombo', 'FixDihedralCombo']
    coord = [bc, ac, dc]
    for i, constr in enumerate([bondcombo, anglecombo, dihedralcombo]):
        atoms.set_constraint()
        atoms.set_constraint(constr)
        atoms.set_positions(atoms.get_positions())
        ClimbFixInternals(atoms,
                          climb_coordinate=[names[i],
                                            [c[:-1] for c in coord[i]]])
