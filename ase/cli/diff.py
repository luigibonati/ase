import numpy as np
from ase.io import read, iread

# there are several packages for pretty printing text tables in python
template = "{title}\n{toprule}\n{header}\n{midrule}\n{body}\n{bottomrule}\n{summary}"
cwidth = 72
tabwidth = 10
fd = {}
fd['title'] = 'Differences in Atomic Positions'
fd['toprule'] = fd['bottomrule'] = '=' * cwidth
fd['midrule'] = '-' * cwidth
fd['header'] = '\t'.join(
    ['index', '   Δx   ', '   Δy   ', '   Δz   ', '   Δ   ', 'Δ rank', 'element'])
row = '{0}\t{1:+.1E}\t{2:+.1E}\t{3:+.1E}\t{4:.1E}\t{5}\t{6}'


def table_gen(elements, positions1, positions2, natural_order=False):
    differences = np.array(positions2) - np.array(positions1)
    magnitude_differences = np.array(
        [np.linalg.norm(diff) for diff in differences])
    pointers = sorted(
        range(
            len(magnitude_differences)),
        key=lambda x: -
        magnitude_differences[x])
    body = []
    if natural_order:
        for c, difference in enumerate(differences):
            body.append(
                row.format(
                    c,
                    *difference,
                    magnitude_differences[c],
                    pointers[c],
                    elements[c]))
    else:  # rank order
        for c, pointer in enumerate(pointers):
            body.append(
                row.format(
                    pointer,
                    *differences[pointer],
                    magnitude_differences[pointer],
                    c,
                    elements[pointer]))
    fd['body'] = '\n'.join(body)
    sm = sum(magnitude_differences**2)
    rmsd = np.sqrt(sm) / len(differences)
    fd['summary'] = 'RMSD = {0:.2f} Å'.format(rmsd)

    return template.format(**fd).expandtabs(tabwidth)

# there are several packages for pretty printing text tables in python
fd2 = {}
fd2['title'] = 'Forces and Energies'
fd2['toprule'] = fd2['bottomrule'] = '=' * cwidth
fd2['midrule'] = '-' * cwidth
fd2['header'] = '\t'.join(
    ['index', '   Fx   ', '   Fy   ', '   Fz   ', '   F   ', 'F rank', 'element'])
row2 = '{0}\t{1:+.1E}\t{2:+.1E}\t{3:+.1E}\t{4:.1E}\t{5}\t{6}'

def table_gen_2(elements, forces, energy, natural_order=False):
    force_ranks = sorted(range(len(elements)),key=lambda x: -np.linalg.norm(forces[x]))
    body = []
    if natural_order:
        for c, force in enumerate(forces):
            body.append(
                row.format(
                    c,
                    *force,
                    np.linalg.norm(force),
                    force_ranks[c],
                    elements[c]
                    ))
    else:  # rank order
        for c, pointer in enumerate(pointers):
            body.append(
                row.format(
                    pointer,
                    *forces[pointer],
                    np.linalg.norm(forces[pointer]),
                    c,
                    elements[pointer]
                    ))

    fd2['body'] = '\n'.join(body)
    fd2['summary'] = 'Energy = {0:.2f} eV'.format(energy)

    return template.format(**fd2).expandtabs(tabwidth)


class CLICommand:
    """Print differences between atoms/calculations.

    Specify more things here in a second block.

    A final block for final specification.
    """

    @staticmethod
    def add_arguments(parser):
        add = parser.add_argument
        add('files', metavar='file(s)', 
                help='either one trajectory file or two non-trajectory files',
                nargs = '+')
        add('-r', '--rank-order', action="store_true", 
                help="Order atoms by rank")
        add('-c', '--calculator-outputs', action="store_true", help="display calculator outputs")

    @staticmethod
    def run(args, parser):
        from ase.io.formats import UnknownFileTypeError
        if len(args.files) == 1 and not args.calculator_outputs: 
            try:
                traj = iread(args.files[0])
            except UnknownFileTypeError:
                raise UnknownFileTypeError

            counter = 0
            for atoms in traj:
                elements = atoms.symbols
                positions = atoms.positions
                if counter > 0:
                    # same number of atoms and order
                    assert((elements == elements0).all())
                    print('\n'*2)
                    print('Iteration/Image Numbers {}-{}'.format(counter+1,counter))
                    print(
                            table_gen(
                                elements,
                                positions0,
                                positions,
                                natural_order = not args.rank_order
                                )
                            )
                positions0 = positions
                elements0 = elements
                counter += 1

        elif len(args.files) == 1 and args.calculator_outputs:
            #relies on the (undocumented) gui Images object
            try:
                from ase.gui.images import Images
                images = Images()
                images.read([args.files[0]])
            except UnknownFileTypeError:
                raise UnknownFileTypeError

            counter = 0
            for atoms in images:
                elements = atoms.symbols
                positions = atoms.positions
                if counter > 0:
                    # same number of atoms and order
                    assert((elements == elements0).all())
                    print('\n'*2)
                    print('Iteration/Image Numbers {}-{}'.format(counter+1,counter))
                    print(
                            table_gen(
                                elements,
                                positions0,
                                positions,
                                natural_order = not args.rank_order
                                )
                            )
                positions0 = positions
                elements0 = elements
                forces = images.get_forces(atoms)
                energy = images.get_energy(atoms)
                print('\n')
                print(
                        table_gen_2(
                            elements,
                            forces,
                            energy,
                            natural_order = not args.rank_order
                            )
                        )
                counter += 1

        elif len(args.files) == 2:
            f1 = args.files[0]
            f2 = args.files[1]
            try:
                atoms1 = read(f1)
                elements = atoms1.symbols
                positions1 = atoms1.positions
            except UnknownFileTypeError:
                raise UnknownFileTypeError
            try:
                atoms2 = read(f2)
                positions2 = atoms2.positions
            except UnknownFileTypeError:
                raise UnknownFileTypeError

            # same number of atoms and order
            assert((atoms2.symbols == elements).all())

            print(
                table_gen(
                    elements,
                    positions1,
                    positions2,
                    natural_order=not args.rank_order))
        else:
            raise Exception('Pass either 1 or 2 arguments')
