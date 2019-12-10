from argparse import RawTextHelpFormatter
from ase.io import read, iread
from ase.gui.images import Images

class CLICommand:
    """Print differences between atoms/calculations.

    Supports taking differences between different calculation runs of the same system as well as neighboring geometric images for one calculation run of a system. For those things which more than the difference is valuable, such as the magnitude of force or energy, the average magnitude between two images or the value for both images is given. Given the difference and the average as x and y, the two values can be calculated as x + y/2 and x - y/2.

    It is possible to fully customize the order of the columns and the sorting of the rows by their column fields. A copy of the template file can be placed in ~/.ase where ~ is the user root directory.
    """

    @staticmethod
    def add_arguments(parser):
        add = parser.add_argument
        parser.formatter_class = RawTextHelpFormatter
        add('files', metavar='file(s)',
            help="""
            2 non-trajectory files: difference between them
            1 trajectory file: difference between consecutive images
            2 trajectory files: difference between corresponding image numbers""",
            nargs='+')
        add('-r', '--rank-order', metavar='field', nargs='?', const='d', type=str,
            help="""order atoms by rank, see --template help for possible fields
            The default value, when specified, is d.
            When not specified, ordering is the same as that provided by the generator.
            For hiearchical sorting, see template.
            """)  # ,formatter_class=RawTextHelpFormatter)
        add('-c', '--calculator-outputs', action="store_true",
            help="display calculator outputs of forces and energy")
        add('-s', '--show-only', metavar='n', type=int,
            help="show only so many lines (atoms) in each table, useful if rank ordering")
        add('-t', '--template', metavar='template', nargs='?', const='rc',
            help="""
        Without argument, looks for ~/.ase/template.py.
        Otherwise, expects the comma separated list of the fields to include in their left-to-right order.
        Optionally, specify the lexicographical sort hierarchy (0 is outermost sort) and if the sort should be ascending or descending (1 or -1).
        By default, sorting is descending, which makes sense for most things except index (and rank, but one can just sort by the thing which is ranked to get ascending ranks).

        example: ase diff start.cif stop.cif --template i:0:1,el,dx,dy,dz,d,rd

        possible fields:
            i: index
            dx,dy,dz,d: displacement/displacement components
            dfx,dfy,dfz,df: difference force/force components
            afx,afy,afz,af: average force/force components (between first and second iteration)
            an: atomic number
            el: atomic element
            t: atom tag
            r<col>: the rank of that atom with respect to the column

        It is possible to change formatters in a template file.
        """)
        add('-l', '--log-file', metavar='logfile', help="print table to file")

    @staticmethod
    def run(args, parser):
        import sys
        # output
        if args.log_file is None:
            out = sys.stdout
        else:
            out = open(args.log_file, 'w')

        # templating
        if args.template is None:
            from ase.cli.template import render_table, render_table_calc, field_specs_on_conditions
            field_specs = field_specs_on_conditions(
                args.calculator_outputs, args.rank_order)
        elif args.template == 'rc':
            import os
            homedir = os.environ['HOME']
            sys.path.insert(0, homedir + '/.ase')
            from templaterc import render_table, render_table_calc, field_specs_on_conditions
            # this has to be named differently because python does not
            # redundantly load packages
            field_specs = field_specs_on_conditions(
                args.calculator_outputs, args.rank_order)
        else:
            field_specs = args.template.split(',')
            if not args.calculator_outputs:
                for field_spec in field_specs:
                    if 'f' in field_spec:
                        raise Exception(
                            'field requiring calculation outputs without --calculator-outputs')

        if len(args.files) == 2:
            f1 = args.files[0]
            f2 = args.files[1]

            images1 = Images()
            images1.read([f1])
            images2 = Images()
            images2.read([f2])
            from math import isnan
            if isnan(images1.get_energy(images1.get_atoms(0))): #assuming trajectory files w/o calculated energies are not being compared
                atoms1 = read(f1)
                atoms2 = read(f2)
                print('images {}-{}'.format(1,0))
                t = render_table(
                    field_specs,
                    atoms1,
                    atoms2,
                    show_only=args.show_only)
                print(t, file=out)
            else:
                if args.calculator_outputs:
                    for counter in range(len(images1)):
                        print('images {}-{}'.format(counter + 1, counter), file=out)
                        t = render_table_calc(
                            field_specs, images1, images2, counter, show_only=args.show_only)
                        print(t, file=out)
                else:
                    for counter in range(len(images1)):
                        print('images {}-{}'.format(counter + 1, counter), file=out)
                        t = render_table(
                            field_specs,
                            images1.get_atoms(counter),
                            images2.get_atoms(counter),
                            show_only=args.show_only)
                        print(t, file=out)

        elif len(args.files) == 1:
            f1 = args.files[0]

            if args.calculator_outputs:
                images = Images()
                images.read([f1])
                for counter in range(len(images) - 1):
                    print('images {}-{}'.format(counter + 1, counter), file=out)
                    t = render_table_calc(
                        field_specs, images, images, counter, show_only=args.show_only)
                    print(t, file=out)
            else:
                traj = iread(f1)
                atoms_list = list(traj)
                for counter in range(len(atoms_list) - 1):
                    t = render_table(field_specs,
                                     atoms_list[counter],
                                     atoms_list[counter + 1],
                                     show_only=args.show_only)
                    print('images {}-{}'.format(counter + 1, counter), file=out)
                    print(t, file=out)
