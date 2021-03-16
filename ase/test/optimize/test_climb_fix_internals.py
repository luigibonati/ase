from ase.build import fcc100, add_adsorbate
from ase.constraints import FixAtoms, FixInternals
from ase.calculators.emt import EMT
from ase.optimize.climbfixinternals import BFGSClimbFixInternals
from ase.vibrations import Vibrations
import numpy as np


def setup_atoms():
    """Setup transition state search for the diffusion barrier for a Pt atom
    on a Pt surface."""
    atoms = fcc100('Pt', size=(2, 2, 1), vacuum=10.0)
    add_adsorbate(atoms, 'Pt', 1.611, 'hollow')
    atoms.set_constraint(FixAtoms(list(range(4))))  # freeze the slab
    return atoms


def test_climb_fix_internals():
    """Climb along the constrained bondcombo coordinate while optimizing the
    remaining degrees of freedom after each climbing step.
    For the definition of constrained internal coordinates see the
    documentation of the classes FixInternals and ClimbFixInternals."""
    atoms = setup_atoms()
    atoms.calc = EMT()

    # Define reaction coordinate via linear combination of bond lengths
    reaction_coord = [[0, 4, 1.0], [1, 4, 1.0]]  # 1 * bond_1 + 1 * bond_2
    # Use current value `FixInternals.get_combo(atoms, reaction_coord)` as initial value
    bondcombo = [None, reaction_coord]  # 'None' will convert to the current value
    atoms.set_constraint([FixInternals(bondcombos=[bondcombo])] + atoms.constraints)

    # Optimizer for transition state search along reaction coordinate
    dyn = BFGSClimbFixInternals(atoms, climb_coordinate=reaction_coord)

    # Converge to a saddle point
    dyn.run(fmax=0.05)

    # Validate transition state by one imaginary vibrational mode
    vib = Vibrations(atoms, indices=[4])
    vib.run()
    assert ((np.imag(vib.get_energies()) > 0) == [True, False, False]).all()
# end example for documentation


def test_initialization_with_different_constraints():
    """Remember to provide reaction coordinates as nested lists.
    Definitions in this example are arbitrary,
    the point is to check whether initialization is successful."""
    atoms = setup_atoms()

    # setup different constraints
    bond = FixInternals(bonds=[[atoms.get_distance(0, 1), [0, 1]]])
    angle = FixInternals(angles_deg=[[atoms.get_angle(0, 1, 2), [0, 1, 2]]])
    dihedral = FixInternals(dihedrals_deg=[[atoms.get_dihedral(0, 1, 2, 3),
                                            [0, 1, 2, 3]]])

    bc = [[0, 4, -1.0], [1, 4, 1.0]]
    ac = [[0, 1, 2, -1.0], [0, 2, 3, 1.0]]
    dc = [[0, 1, 2, 3, -1.0], [0, 1, 4, 3, 1.0]]

    value = FixInternals.get_combo(atoms, bc)
    bondcombo = FixInternals(bondcombos=[[value, bc]])

    value = FixInternals.get_combo(atoms, ac)
    anglecombo = FixInternals(anglecombos=[[value, ac]])

    value = FixInternals.get_combo(atoms, dc)
    dihedralcombo = FixInternals(dihedralcombos=[[value, dc]])

    # test initialization of BFGSClimbFixInternals with different constraints
    coord = [[0, 1], [0, 1, 2], [0, 1, 2, 3], bc, ac, dc]
    for i, constr in enumerate([bond, angle, dihedral,
                                bondcombo, anglecombo, dihedralcombo]):
        atoms.set_constraint(constr)
        BFGSClimbFixInternals(atoms, climb_coordinate=coord[i])
