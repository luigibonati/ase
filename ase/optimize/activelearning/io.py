from ase import io
from ase.parallel import parallel_function, parprint
import copy


@parallel_function
def dump_observation(atoms, filename, restart):
    """
    Saves a trajectory file containing the atoms observations.

    Parameters
    ----------
    atoms: object
        Atoms object to be appended to previous observations.
    filename: string
        Name of the trajectory file to save the observations.
    restart: boolean
        Append mode (true or false).
     """

    if restart is True:
        try:
            prev_atoms = io.read(filename, ':')  # Actively searching.
            if atoms not in prev_atoms:  # Avoid duplicates.
                parprint('Updating observations...')
                new_atoms = prev_atoms + [atoms]
                io.write(filename=filename, images=new_atoms)
        except Exception:
            io.write(filename=filename, images=atoms, append=True)
    if restart is False:
        io.write(filename=filename, images=atoms, append=False)


@parallel_function
def attach_calculator(train_images, test_images, calculator):
    """
    Create duplicate of a calculator which contained a trained model and
    appends this calculator to different Atoms objects. This avoids training
    the same process for multiple images that contain the same training
    observations.

    Parameters
    ----------
    train_images: list
        List of Atoms containing the observations to build the model.
    test_images: list
        List of Atoms to be tested.
    calculator: object
        Calculator to be appended to the different test images. This is
        usually the calculator containing the model (e.g. a GPCalculator).

    Returns
    -------
    test_images: list
        List of Atoms with a model calculator attached.

    """
    calc = copy.deepcopy(calculator)
    calc.update_train_data(train_images)

    for i in test_images:
        i.set_calculator(copy.deepcopy(calc))
        i.get_potential_energy()
    return test_images
