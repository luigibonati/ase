import numpy as np
from ase.parallel import parallel_function
from ase import io
from ase.io.trajectory import TrajectoryWriter
import os


@parallel_function
def dump_observation(atoms, filename, restart, method='-'):
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
    method: string
        Label with the optimizer name to be appended in atoms.info['method'].
     """
    atoms.info['method'] = method

    if restart:
        if os.path.exists(filename):
            prev_atoms = io.read(filename, ':')  # Active learning.
            if atoms not in prev_atoms:  # Avoid duplicates.
                # Update observations.
                trj = TrajectoryWriter(atoms=atoms, filename=filename,
                                       mode='a')
                trj.write()
        else:
            io.write(filename=filename, images=atoms, append=False)
    if not restart:
        io.write(filename=filename, images=atoms, append=False)


@parallel_function
def get_fmax(atoms):
    """
    Returns fmax for a given atoms structure.
    """
    forces = atoms.get_forces()
    return np.sqrt((forces**2).sum(axis=1).max())


