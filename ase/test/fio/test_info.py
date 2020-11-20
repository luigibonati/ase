from ase import Atoms
from ase.io import Trajectory


def test_info():
    # Create a molecule with an info attribute
    info = dict(creation_date='2011-06-27',
                chemical_name='Hydrogen',
                # custom classes also works provided that it is
                # imported and pickleable...
                foo={'seven': 7})

    molecule = Atoms('H2', positions=[(0., 0., 0.), (0., 0., 1.1)], info=info)
    assert molecule.info == info

    atoms = molecule.copy()
    assert atoms.info == info

    with Trajectory('info.traj', 'w', atoms=molecule) as traj:
        traj.write()

    with Trajectory('info.traj') as t:
        atoms = t[-1]

    print(atoms.info)
    assert atoms.info == info
