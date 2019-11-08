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


class TrainingSet:

    """
    Beautiful documentation comes in here.
    """

    def __init__(self, destination, use_previous_observations):
        if type(destination) is not str:
            raise NotImplementedError("*destination* should be a file")
        if not destination.endswith('.traj'):
            raise NotImplementedError("*destination* should be a trajectory file")

        self.use_prev_obs = use_previous_observations
        self.destination = destination

    def dump(self, atoms, method):
        dump_observation(atoms, filename = self.destination,
                         method = method,
                         restart = self.use_previous_obs)
    def load_set(self):
        return io.read(self.destination, ':')

    def load_last(self):
        return io.read(self.destination, -1)


@parallel_function
def get_fmax(atoms):
    """
    Returns fmax for a given atoms structure.
    """
    forces = atoms.get_forces()
    return np.sqrt((forces**2).sum(axis=1).max())
