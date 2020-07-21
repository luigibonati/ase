import os
import json
from warnings import warn
from copy import deepcopy
from typing import List, Any, IO

import numpy as np

from ase import Atoms
from ase.io.jsonio import read_json, write_json
from ase.units import Hartree, Bohr
from ase.calculators.singlepoint import SinglePointCalculator
from ase.utils import reader


def write_bagel_in(
    fd: IO,
    atoms: Atoms,
    properties: List[str] = None,
    **params: Any
) -> None:
    params = deepcopy(params)
    if properties is None:
        properties = ['energy']

    # ASE-specific keyword for determining which energy to use from ENERGY.out
    # and which FORCE_{state}.out to use.
    params.pop('state', None)

    inp = params.pop('raw_json', {'bagel': []})
    bgl = inp['bagel']
    if bgl and isinstance(bgl[0], dict) and bgl[0].get('title') == 'molecule':
        mol = bgl[0]
    else:
        mol = dict(title='molecule')
        bgl.insert(0, mol)

    # force the use of Angstrom position units
    mol['angstrom'] = True
    mol['geometry'] = []
    for atom in atoms:
        mol['geometry'].append({
            'atom': atom.symbol,
            'xyz': tuple(atom.position)
        })

    for key, val in params.items():
        # assume any non-dict top-level kwargs go in the 'molecule' block
        if not isinstance(val, dict):
            mol[key] = val
            continue

        for entry in bgl:
            if isinstance(entry, dict) and entry.get('title') == key:
                entry.update(val)
        else:
            val['title'] = key
            bgl.append(val)

    write_json(fd, inp)


def read_json_in(fd: IO) -> Atoms:
    inp = read_json(fd)
    for entry in inp['bagel']:
        if isinstance(entry, dict) and entry['title'] == 'molecule':
            break
    else:
        raise IOError("No valid molecule block found!")

    symbols = [atom['atom'] for atom in entry]
    pos = np.array([atom['xyz'] for atom in entry])
    if not entry.get('angstrom', False):
        pos *= Bohr
    return Atoms(symbols, positions=pos)


def _extract_geom(fd: IO) -> Atoms:
    symbols = []
    positions = []
    for line in fd:
        try:
            atom = json.loads(line.strip().rstrip(','))
        except json.JSONDecodeError:
            break
        symbols.append(atom['atom'])
        positions.append(atom['xyz'])
    return Atoms(symbols, positions=np.array(positions) * Bohr)


def read_bagel_out(fd: IO) -> Atoms:
    atoms = None
    energy = None
    forces = None
    e_scf = None
    for line in fd:
        if line.strip() == '*** Geometry ***':
            fd.readline()
            atoms = _extract_geom(fd)
        elif line.strip().startswith("o Fock build"):
            # Annoyingly, Bagel does not summarize the final converged
            # energy once it reaches convergence, so we have to parse
            # all intermediate energies during SCF.
            #
            # To avoid accidentally returning an unconverged energy,
            # we store the intermediate energies in `e_scf` and transfer
            # to `energy` only once SCF convergence has been established.
            e_scf = float(fd.readline().split()[1]) * Hartree
        elif line.strip() == '* SCF iteration converged.':
            energy = e_scf
            e_scf = None
        elif line.strip().startswith('MP2 total energy:'):
            energy = float(line.split()[3]) * Hartree
        elif line.strip() == '* Nuclear energy gradient':
            forces = []
            fd.readline()
            while fd.readline().strip().startswith('o Atom'):
                forces.append([
                    float(fd.readline().split()[1]) for _ in range(3)
                ])
            forces = -np.array(forces) * Hartree / Bohr
    if atoms is None:
        raise IOError("Failed to locate molecule geometry!")
    if energy is not None or forces is not None:
        atoms.calc = SinglePointCalculator(atoms, energy=energy, forces=forces)
    return atoms


@reader
def read_bagel_energy(fd: IO, state: int = 0) -> float:
    return float(fd.readlines()[state]) * Hartree


@reader
def read_bagel_forces(fd: IO) -> np.ndarray:
    forces = []
    fd.readline()
    for line in fd:
        entry = line.split()
        if not entry:
            break
        forces.append(tuple(map(float, entry[1:])))
    return -np.array(forces) * Hartree / Bohr


@reader
def get_bagel_results(fd: IO, directory: str, state: int = 0) -> Atoms:
    atoms = read_bagel_out(fd)
    efile = os.path.join(directory, 'ENERGY.out')
    try:
        energy = read_bagel_energy(efile, state=state)
    except FileNotFoundError:
        warn(
            'BAGEL ENERGY.out file not found. Reported energies and forces '
            'may be inaccurate. Please set export=True in the relevant '
            'BAGEL input sections.'
        )
        return atoms
    ffile = os.path.join(directory, 'FORCE_{}.out'.format(state))
    try:
        forces = read_bagel_forces(ffile)
    except FileNotFoundError:
        forces = None
    atoms.calc = SinglePointCalculator(atoms, energy=energy, forces=forces)
    return atoms
