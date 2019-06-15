from ase import io
from ase.parallel import parallel_function, rank, parprint
import numpy as np
import datetime

@parallel_function
def dump_experience(atoms, filename):
    filename = filename.split('.')[0] + '_experiences.traj'
    try:
        prev_atoms = io.read(filename, ':')
        if atoms not in prev_atoms:  # Avoid duplicates.
            parprint('Updating images (experiences) pool...')
            new_atoms = prev_atoms + [atoms]
            io.write(filename=filename, images=new_atoms)
    except Exception:
        io.write(filename=filename, images=atoms)  # Create new atoms pool.


@parallel_function
def get_fmax(atoms):
    """
    Returns fmax for a given atoms structure.
    """
    forces = atoms.get_forces()
    return np.sqrt((forces**2).sum(axis=1).max())


@parallel_function
def print_time():
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")


@parallel_function
def print_cite_neb():
    msg = "-----------------------------------------------------------"
    msg += "-----------------------------------------------------------\n"
    msg += "You are using GPNEB. Please cite: \n"
    msg += "[1] J. A. Garrido Torres, M. H. Hansen, P. C. Jennings, "
    msg += "J. R. Boes and T. Bligaard. Phys. Rev. Lett. 122, 156001. "
    msg += "https://doi.org/10.1103/PhysRevLett.122.156001 \n"
    msg += "[2] O. Koistinen, F. B. Dagbjartsdottir, V. Asgeirsson, A. Vehtari"
    msg += " and H. Jonsson. J. Chem. Phys. 147, 152720. "
    msg += "https://doi.org/10.1063/1.4986787 \n"
    msg += "[3] E. Garijo del Rio, J. J. Mortensen and K. W. Jacobsen. "
    msg += "arXiv:1808.08588. https://arxiv.org/abs/1808.08588v1. \n"
    msg += "-----------------------------------------------------------"
    msg += "-----------------------------------------------------------"
    parprint(msg)

