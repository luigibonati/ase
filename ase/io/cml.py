"""Module to read atoms in chemical json file format.

https://wiki.openchemistry.org/Chemical_JSON
"""
import json
import numpy as np

from ase import Atoms, Atom
from ase.data import chemical_symbols
from ase.cell import Cell


def read_cml(fileobj):
    data = json.load(fileobj)
    atoms = Atoms()
    datoms = data['atoms']

    for i in datoms['elements']['number']:
        atoms.append(Atom(chemical_symbols[i]))

    if 'unit cell' in data:
        cell = data['unit cell']
        a = cell['a']
        b = cell['b']
        c = cell['c']
        alpha = cell['alpha']
        beta = cell['beta']
        gamma = cell['gamma']
        atoms.cell = Cell.fromcellpar([a, b, c, alpha, beta, gamma])
    
    coords = datoms['coords']
    if '3d' in coords:
        positions = np.array(coords['3d']).reshape(len(atoms), 3)
        atoms.set_positions(positions)
    else:
        positions = np.array(coords['3d fractional']).reshape(len(atoms), 3)
        atoms.set_scaled_positions(positions)
        
    yield atoms
