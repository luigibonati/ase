import pytest
import re

import ase.build
from ase.io import write, read


# create atom with custom mass - from a list of positions or using ase.build.molecule
def write_read_atom(atom, tmp_path):
    write("{0}/{1}".format(tmp_path, "castep_test.cell"), atom)
    return read("{0}/{1}".format(tmp_path, "castep_test.cell"))


# write to .cell and check that .cell has correct species_mass block in it
@pytest.mark.parametrize(
    "mol, custom_masses, expected_species, expected_mass_block",
    [
        ("CH4", {2: [1]}, ["C", "H:0", "H", "H", "H"], ["H:0 2.0"]),
        ("CH4", {2: [1, 2, 3, 4]}, ["C", "H", "H", "H", "H"], ["H 2.0"]),
        ("C2H5", {2: [2, 3]}, ["C", "C", "H:0", "H:0", "H", "H", "H"], ["H:0 2.0"]),
        (
            "C2H5",
            {2: [2], 3: [3]},
            ["C", "C", "H:0", "H:1", "H", "H", "H"],
            ["H:0 2.0", "H:1 3.0"],
        ),
    ],
)
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_custom_mass_write(
    mol, custom_masses, expected_species, expected_mass_block, tmp_path
):

    custom_atom = ase.build.molecule(mol)
    atom_positions = custom_atom.positions

    for mass, pos_list in custom_masses.items():
        for pos in pos_list:
            custom_atom[pos].mass = mass

    atom_masses = custom_atom.get_masses()
    new_atom = write_read_atom(custom_atom, tmp_path)

    # check atom has been written and read correctly
    assert (atom_positions == new_atom.positions).all()
    assert (atom_masses == new_atom.get_masses()).all()

    # check that file contains appropriate blocks
    with open("{0}/{1}".format(tmp_path, "castep_test.cell"), "r") as f:
        data = f.read().replace("\n", "\\n")

    position_block = re.search(r"%BLOCK POSITIONS_ABS.*%ENDBLOCK POSITIONS_ABS", data)
    assert position_block

    pos = position_block.group().split("\\n")[1:-1]
    species = [p.split(" ")[0] for p in pos]
    assert species == expected_species

    mass_block = re.search(r"%BLOCK SPECIES_MASS.*%ENDBLOCK SPECIES_MASS", data)
    assert mass_block

    masses = mass_block.group().split("\\n")[1:-1]
    assert masses == expected_mass_block


# test setting a custom species on different atom before write
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_custom_mass_overwrite(tmp_path):
    custom_atom = ase.build.molecule("CH4")
    custom_atom[1].mass = 2
    atoms = write_read_atom(custom_atom, tmp_path)

    # test that changing masses when custom masses defined causes errors
    atoms[3].mass = 3
    with pytest.raises(ValueError) as e:
        atoms.write("{0}/{1}".format(tmp_path, "castep_test2.cell"))

    print(e)
    assert (
        "Could not write custom mass block for H."
        == e.value.args[0].split("\n")[0].strip()
    )
