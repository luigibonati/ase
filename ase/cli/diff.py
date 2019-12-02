import numpy as np
from ase.io import read

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


class CLICommand:
    """Print/plot differences between atoms/calculations.

    Specify more things here in a second block.

    A final block for final specification.
    """

    @staticmethod
    def add_arguments(parser):
        add = parser.add_argument
        add('f1', metavar='initial file', help='initial file')
        add('f2', metavar='final file', help='final file')
        add('-e', '--energies', action="store_true", help="Plot energies")
        add('-d', '--displacements', action="store_true", help="Plot displacements")
        add('-f', '--forces', action="store_true", help="Plot forces")
        add('-r', '--rank-order', action="store_true", help="Order atoms by rank")

    @staticmethod
    def run(args, parser):
        from ase.io.formats import UnknownFileTypeError
        try:
            atoms1 = read(args.f1)
            elements = atoms1.symbols
            positions1 = atoms1.positions
        except UnknownFileTypeError:
            raise UnknownFileTypeError
        try:
            atoms2 = read(args.f2)
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
