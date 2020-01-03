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
            from ase.cli.template import render_table, field_specs_on_conditions, rmsd, energy_delta
            field_specs = field_specs_on_conditions(
                args.calculator_outputs, args.rank_order)
        elif args.template == 'rc':
            import os
            homedir = os.environ['HOME']
            sys.path.insert(0, homedir + '/.ase')
            from templaterc import render_table, field_specs_on_conditions, rmsd, energy_delta
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

        two_files = len(args.files) == 2

        f1 = args.files[0]
        a1 = read(f1, index = ':')
        l1 = len(a1)

        if two_files:
            f2 = args.files[1]
            a2 = read(f2, index = ':')
            l2 = len(a2)
            same_length = l1 == l2
            one_l_one = l1 or l2
        
            if not same_length and not one_l_one:
                raise Exception("Trajectory files are not the same length and both > 1")
            elif not same_length and one_l_one:
                print("""One file contains one image and the other multiple images, 
                    assuming you want to compare all images with one reference image""")
                if l1 > l2:
                    a2 = l1*a2
                else:
                    a1 = l2*a1
                def header_fmt(c):
                    return 'sys-ref image # {}'.format(c)
            else:
                def header_fmt(c):
                    return 'sys2-sys1 image # {}'.format(c)
        else:
            a2 = a1.copy()
            a1 = a1[:-1]
            a2 = a2[1:]
            def header_fmt(c):
                 return 'images {}-{}'.format(c+1,c)

        from math import isnan
        if a1[0].get_calculator() == None:
            has_calc = False
        else: 
            has_calc = True


        pairs = zip(a1, a2)
        if has_calc and args.calculator_outputs:
            for counter, pair in enumerate(pairs):
                print(header_fmt(counter), file=out)
                t = render_table(
                    field_specs, *pair, show_only=args.show_only, summary_function = rmsd)
                t += '\n'
                t += render_table(
                    field_specs, *pair, show_only=args.show_only, summary_function = energy_delta)
                print(t, file=out)
        else:
            for counter,pair in enumerate(pairs):
                print(header_fmt(counter), file=out)
                t = render_table(
                    field_specs, *pair, show_only=args.show_only, summary_function = rmsd)
                print(t, file=out)
