from ase import Atoms
from ase.constraints import FixInternals
from ase.calculators.emt import EMT
from ase.optimize.climbfixinternals import ClimbFixInternals
from ase.visualize import view
from ase.io import read

def get_bondcombo(atoms, bondcombo):
    """Returns current value of linear combination of bonds defined via
    bondcombo (see FixInternals)."""
    return sum([defin[2] * atoms.get_distance(*defin[0:2]) for
                defin in bondcombo])

def test_initialization_with_different_constraints():
    pass

def test_climb_fix_internals():
    """Find transition state for SN2 reaction:
    Substitution of MeCu by Cu (Cu, because Cl is not supported by EMT()."""
    positions = [[7.07784622, 6.36737904, 8.47511833],
                 [7.50617774, 5.97274016, 7.55045035],
                 [6.00151453, 6.51727226, 8.36050116],
                 [7.56351536, 7.30754202, 8.74801782],
                 [6.58494379, 8.43777561, 6.21601465],
                 [7.36058358, 5.14853782, 9.81663707]]
    atoms = Atoms(symbols='CH3Cu2', positions=positions)

    # Define reaction coordinate via linear combination of bond lengths
    reaction_coord = [[0, 4, -1.0], [0, 5, 1.0]]  # = 1 * bond_1 -1 * bond_2
    bondcombo = [get_bondcombo(atoms, reaction_coord), reaction_coord]

    atoms.set_constraint(FixInternals(bondcombos=[bondcombo]))

    # Calculate using EMT
    atoms.calc = EMT()
    atoms.get_potential_energy()

    # Set the optimizer
    dyn = ClimbFixInternals(atoms,
                            climb_coordinate=['FixBondCombo',
                                              [[0, 4], [0, 5]]],
                            trajectory='opt.traj')
    
    # Converge to a saddle point
    dyn.run(fmax=0.01)
    #a = read('opt.traj', index=':')

    # Test the results
    assert(abs(get_bondcombo(atoms, reaction_coord)) < 0.003)
