import numpy as np
from ase.io import read, iread

# there are several packages for pretty printing text tables in python
# pandas would make this very easy with to_csv
# Cheetah templates allow for loops for variable number of table rows
template = "{title}\n{toprule}\n{header}\n{midrule}\n{body}\n{bottomrule}\n{summary}"
cwidth = 72
tabwidth = 9
fd = {}
fd['title'] = 'Differences in Atomic Positions'
fd['toprule'] = fd['bottomrule'] = '=' * cwidth
fd['midrule'] = '-' * cwidth
fmt = "{:^" + str(tabwidth) + "}"
header = ['index', 'Δx', 'Δy', 'Δz', 'Δ', 'Δ rank', 'element']
fd['header'] = ''.join(fmt * len(header)).format(*header)
fd['row'] = '{0:^9}{1:^+9.1E}{2:^+9.1E}{3:^+9.1E}{4:^9.1E}{5:^9}{6:^9}'


def table_gen(
        elements,
        positions1,
        positions2,
        order_by=None,
        format_dictionary=fd,
        show_only=None):
    differences = np.array(positions2) - np.array(positions1)
    magnitude_differences = np.array(
        [np.linalg.norm(diff) for diff in differences])
    n = len(differences)
    pointers = sorted(range(n), key=lambda x:-magnitude_differences[x])
    if show_only is not None:
        cmax = show_only
    else:
        cmax = len(differences)

    fmt = fd['row'].format
    c = 0
    body = []

    if order_by is None:
        while c < cmax:
            body.append(fmt(c,
                            *differences[c],
                            magnitude_differences[c],
                            pointers[c],
                            elements[c]))
            c += 1
    elif order_by == 'delta':  # rank order
        while c < cmax:
            body.append(fmt(pointers[c],
                            *differences[pointers[c]],
                            magnitude_differences[pointers[c]],
                            c,
                            elements[pointers[c]]))
            c += 1
    else:
        raise Exception(f'do not support ordering by {order_by}')

    format_dictionary['body'] = '\n'.join(body)
    sm = sum(magnitude_differences**2)
    rmsd = np.sqrt(sm) / len(differences)
    format_dictionary['summary'] = 'RMSD = {0:.2f} Å'.format(rmsd)

    return template.format(**format_dictionary)

#fd2 = {}
#fd2['title'] = 'Forces and Energies'
#fd2['toprule'] = fd2['bottomrule'] = '=' * cwidth
#fd2['midrule'] = '-' * cwidth
# fd2['header'] = '\t'.join(
#    ['index', '   Fx   ', '   Fy   ', '   Fz   ', '   F   ', 'F rank', 'element'])
#row2 = '{0}\t{1:+.1E}\t{2:+.1E}\t{3:+.1E}\t{4:.1E}\t{5}\t{6}'

fd3 = {}
fd3['title'] = 'Displacements, forces, and energies'
fd3['toprule'] = fd3['bottomrule'] = '=' * cwidth
fd3['midrule'] = '-' * cwidth
fmt = "{:^" + str(tabwidth) + "}"
header = ['index', 'Δ', 'Δ rank', 'F', 'F rank', 'ΔF', 'ΔF rank', 'element']
fd3['header'] = ''.join(fmt * len(header)).format(*header)

def table_gen_calc(
        atomic_numbers,
        positions1,
        positions2,
        forces1,
        forces2,
        energy1,
        energy2,
        order_by=None,
        show_only=None,
        verbose=False,
        counter=1):
    # sorting
    n = len(atomic_numbers)
    keys = [range(n),
            atomic_numbers,
            np.linalg.norm(positions2 - positions1, axis=1),
            np.linalg.norm(forces2 - forces1, axis=1),
            np.linalg.norm((forces2 + forces1) / 2, axis=1)]

    nkeys = []
    for key in keys[1:]:
        nkeys.append(np.argsort(key))
    keys += nkeys

    if order_by == 'atomic number':
        inds = keys[5]
    elif order_by == 'delta':
        inds = keys[6]
    elif order_by == 'force delta':
        inds = keys[7]
    elif order_by == 'force average':
        inds = keys[8]
    else:
        inds = keys[0]

    t = np.transpose(keys)
    t = t[inds]
    from ase.data import chemical_symbols
    body = []
    if show_only != None:
        t=t[:show_only]

    if verbose:
        pass #may support adding more columns, 9 is already many

    for row in t:
        i, an, D, DF, AF, _, RD, RDF, RAF = row
        el = chemical_symbols[int(an)]
        #fstrings require python >= 3.6
        if verbose:
#            row = f"{i:^9.0n}{D:^+9.1E}{RD:^9.0n}{DF:^+9.1E}{RDF:^9.0n}{AF:^+9.1E}{RAF:^9.0n}{el:^9}"
            pass
        else:
#            row = f"{i:^9.0n}{D:^+9.1E}{RD:^9.0n}{DF:^+9.1E}{RDF:^9.0n}{AF:^+9.1E}{RAF:^9.0n}{el:^9}"
            fmt = "{:^9.0n}{:^+9.1E}{:^9.0n}{:^+9.1E}{:^9.0n}{:^+9.1E}{:^9.0n}{:^9}".format
            row = fmt(i,D,RD,DF,RDF,AF,RAF,el)
            
        body.append(row)
    fd3['body'] = '\n'.join(body)
    fd3['summary'] = f"E{counter} = {energy1:+.1E}, E{counter+1}  = {energy2:+.1E}, ΔE = {energy2-energy1:+.1E}"
    return template.format(**fd3)


class CLICommand:
    """Print differences between atoms/calculations.

    For those things which more than the difference is valuable, such as the magnitude of force, the average magnitude between two images is given. Given the difference and the average as x and y, the two values can be calculated as x + y/2 and x - y/2.

    A final block for final specification.
    """

    @staticmethod
    def add_arguments(parser):
        add = parser.add_argument
        add('files', metavar='file(s)',
            help='either one trajectory file or two non-trajectory files',
            nargs='+')
        add('--rank-order', nargs='?', const='delta',
            type=str, help="""order atoms by rank, possible ranks are
            atomic number: self-explanatory
            delta: change in position 
            force delta: change in force
            force average: average force in two iterations

            default value, when specified, is delta
            when not specified, ordering is the same as that provided by the generator
            """)
        add('-c', '--calculator-outputs', action="store_true",
            help="display calculator outputs of forces and energy")
        add('-v', '--verbose', action="store_true",
            help="verbosity, whether to show the components of position or force. Note the tables can get very wide.")
        add('-s', '--show-only', type=int, help="show only so many lines (atoms) in each table, useful if rank ordering")

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
                    print('\n' * 2)
                    print('Iteration/Image Numbers {}-{}'.format(counter + 1, counter))
                    print(table_gen(
                        elements,
                        positions0,
                        positions,
                        order_by=args.rank_order,
                        show_only=args.show_only
                    ))
                positions0 = positions
                elements0 = elements
                counter += 1

        elif len(args.files) == 1 and args.calculator_outputs and not args.verbose:
            # relies on the (undocumented) gui Images object
            try:
                from ase.gui.images import Images
                images = Images()
                images.read([args.files[0]])
            except UnknownFileTypeError:
                raise UnknownFileTypeError

            counter = 0
            for atoms in images:
                atomic_numbers = atoms.numbers
                positions = atoms.positions
                forces = images.get_forces(atoms)
                energy = images.get_energy(atoms)
                if counter > 0:
                    # same number of atoms and order
                    assert((atomic_numbers == atomic_numbers0).all())
                    print('\n' * 2)
                    print('Iteration/Image Numbers {}-{}'.format(counter + 1, counter))
                    print(
                        table_gen_calc(atomic_numbers,
                                    positions0,
                                    positions,
                                    forces0,
                                    forces,
                                    energy0,
                                    energy,
                                    order_by=args.rank_order,
                                    show_only=args.show_only,
                                    counter=counter))
                positions0 = positions
                atomic_numbers0 = atomic_numbers
                forces0 = forces
                energy0 = energy
                counter += 1

        elif len(args.files) == 2:
            f1 = args.files[0]
            f2 = args.files[1]
            try:
                atoms1 = read(f1)
                elements = atoms1.symbols
                positions1 = atoms1.positions
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
                    order_by=args.rank_order,
                    show_only=args.show_only))
        else:
            raise Exception('Pass either 1 or 2 positional arguments for files')


if __name__ == "__main__":
    #tests
    import numpy as np
    from ase.gui.images import Images
    images = Images()
    images.read(['vasprun.xml'])
    atomic_numbers = images[0].numbers
    positions1 = images[0].positions
    positions2 = images[1].positions
    forces1 = images.get_forces(images[0])
    forces2 = images.get_forces(images[1])
    energy = images.get_energy(images[0])
    t = table_gen_calc(
        atomic_numbers,
        positions1,
        positions2,
        forces1,
        forces2,
        energy)
    print(t)
