from ase.atoms import Atoms


def read_wout(fileobj, include_wannier_function_centers=True):
    lines = fileobj.readlines()
    return Atoms()
