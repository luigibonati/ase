from ase.io import read, write
from scipy.spatial.distance import euclidean
import numpy as np
from ase.optimize.activelearning.io import get_fmax


class CleanObservations:
    def __init__(self, trajectory,
                 output_trajectory='clean_observations.traj',
                 distance_threshold=0.1, fmax_threshold=0.05,
                 remove_outliers=True):
        """ Sparse observations. Reduce the number of observations by
        removing the observations that are too close to each other.
        The distance_threshold defines the distance in which the
        observations are too close. This class can remove outliers, i.e. Atoms
        images with high energy compared to the other Atoms observations.

        Parameters
        --------------
        trajectory: Trajectory file (in ASE format).
            Filename for the trajectory containing the Atoms images with
            the raw observations (before cleaning).

        output_trajectory: Trajectory file (in ASE format).
            Filename for the trajectory in which the clean structures will
            be saved.

        distance_threshold: float
            Distance threshold delimiting whether two Atoms
            Cartesian coordinates are too close to each other.

        remove_outliers: bool
            If True will remove the images that satisfy the outlier criterion:
            Energy > median_e + 2.5 * mad_e (non-normally distribution).

        fmax_threshold: float
            Structures with fmax below fmax_threshold will be added to the
            clean data set (if not duplicated).

        """
        self.trajectory = trajectory
        self.distance_threshold = distance_threshold
        self.remove_outliers = remove_outliers
        self.output_trajectory = output_trajectory
        self.fmax_threshold = fmax_threshold

    def clean(self):
        """Start the cleaning process. Returns the list of cleaned
           structures."""
        all_structures = read(self.trajectory, ':')
        all_energies = []
        all_fmax = []
        clean_structures = []

        # Start by including all the structures with fmax below fmax_threshold.
        for structure in all_structures:
            if get_fmax(structure) <= self.fmax_threshold:
                is_good = True
                for clean_structure in clean_structures:
                    too_close = images_too_close(image1=structure,
                                                 image2=clean_structure,
                                                 distance_threshold=self.distance_threshold)
                    if too_close is True:
                        is_good = False
                if is_good is True:
                    clean_structures += [structure]
                    print('Converged structure found...')

        # Remove outliers (optional).
        if self.remove_outliers is True:
            structures_without_outliers = []
            for structure in all_structures:
                all_energies += [structure.get_potential_energy()]
                all_fmax += [get_fmax(structure)]

            median_e = np.median(all_energies)
            mad_e = mad(all_energies)

            for e_index in range(0, len(all_structures)):
                if all_energies[e_index] < median_e + 2.5 * mad_e:
                    structures_without_outliers += [all_structures[e_index]]
            all_structures = structures_without_outliers

        # Get clean list of atoms:
        for structure in all_structures:
            all_energies += [structure.get_potential_energy()]
            is_good = True
            for clean_structure in clean_structures:
                too_close = images_too_close(image1=structure,
                                             image2=clean_structure,
                                             distance_threshold=self.distance_threshold)
                if too_close is True:
                    is_good = False
            if is_good is True:
                clean_structures += [structure]
            else:
                print('Atoms structure too close to a previous structure.')

        write(self.output_trajectory, clean_structures)
        return clean_structures


def images_too_close(image1, image2, distance_threshold=0.1):
    """Function to decide whether two structures are too close to each
    other (defined by the distance threshold).
    """
    pos1 = image1.positions.reshape(-1)
    pos2 = image2.positions.reshape(-1)
    d_s1_s2 = euclidean(pos1, pos2)
    if d_s1_s2 < distance_threshold:
        return True
    else:
        return False


def mad(arr):
    """ Median Absolute Deviation:
        https://en.wikipedia.org/wiki/Median_absolute_deviation
    """
    arr = np.ma.array(arr).compressed()
    med = np.median(arr)
    return np.median(np.abs(arr - med))
