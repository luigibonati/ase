# Lattice type for each space group (see ase.spacegroup.Spacegroup)
# stored as a string whose nth element is Spacegroup(n).symbol[0]
spacegroup_centering = ('ØPPPPCPPCCPPCPPCPPPPCCFIIPPPPPPPPPPCCCAAAAFFIIIPP'
                        'PPPPPPPPPPPPPPCCCCCCFFIIIIPPPPIIPIPPPPIIPPPPPPPPI'
                        'IPPPPPPPPIIIIPPPPPPPPIIIIPPPPPPPPPPPPPPPPIIIIPPPR'
                        'PRPPPPPPRPPPPRRPPPPRRPPPPPPPPPPPPPPPPPPPPPPPPPPPP'
                        'FIPIPPFFIPIPPFFIPPIPFIPFIPPPPFFFFII')


# Maps (centering, lattice_system) -> short name of bravais lattice
bravais_classification = {('P', 'triclinic'): 'tri',
                          ('P', 'monoclinic'): 'mcl',
                          ('C', 'monoclinic'): 'mclc',
                          ('P', 'orthorhombic'): 'orc',
                          ('C', 'orthorhombic'): 'orcc',
                          ('A', 'orthorhombic'): 'orcc',  # same as C above
                          ('I', 'orthorhombic'): 'orci',
                          ('F', 'orthorhombic'): 'orcf',
                          ('P', 'tetragonal'): 'tet',
                          ('I', 'tetragonal'): 'bct',
                          ('P', 'rhombohedral'): 'rhl',
                          ('R', 'rhombohedral'): 'rhl',  # same as P above
                          ('P', 'hexagonal'): 'hex',
                          ('P', 'cubic'): 'cub',
                          ('I', 'cubic'): 'bcc',
                          ('F', 'cubic'): 'fcc'}


def get_crystal_system(sg):
    sg = int(sg)
    if sg < 1:
        raise ValueError('Spacegroup must be positive, but is {}'.format(sg))
    if sg < 3:
        return 'triclinic'
    if sg < 16:
        return 'monoclinic'
    if sg < 75:
        return 'orthorhombic'
    if sg < 143:
        return 'tetragonal'
    if sg < 168:
        return 'trigonal'
    if sg < 195:
        return 'hexagonal'
    if sg < 231:
        return 'cubic'
    raise ValueError('Bad spacegroup', sg)


# The rhombohedral spacegroups.
# The point of storing this is that trigonal systems are split into
# rhombohedral and hexagonal lattices
rhl_spacegroups = {146, 148, 155, 160, 161, 166, 167}


def get_lattice_system(sg):
    sg = int(sg)
    cs = get_crystal_system(sg)
    if cs == 'trigonal':
        return 'rhombohedral' if sg in rhl_spacegroups else 'hexagonal'
    else:
        return cs


def get_bravais_lattice(sg):
    sg = int(sg)
    assert sg in range(1, 231)
    sym = spacegroup_centering[sg]
    ls = get_lattice_system(sg)
    return bravais_classification[sym, ls]
