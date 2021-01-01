from ase.io import read


class CLICommand:
    """ Execute code on files.

    The given python code is evaluated on the Atoms object read from
    the input file for each frame of the file. `atoms` is used to 
    denote the Atoms object. Either of -e or -E option should provided
    for evaluating code given as a string or from a file, respectively.
    """

    @staticmethod
    def add_arguments(parser):
        add = parser.add_argument
        add('input', nargs='+', metavar='input-file')
        add('-e', '--exec-code',
            help='Python code to execute on each atoms. The Atoms'
            ' object is available as `atoms`. '
            'Example: For printing cell parameters from all the '
            'frames, `print(atoms.cell.cellpar())`')
        add('-E', '--exec-file',
            help='Python source code file to execute on each '
            'frame, usage is as for -e/--exec-code.')
        add('-i', '--input-format', metavar='FORMAT',
            help='Specify input FORMAT')
        add('-n', '--image-number',
            default=':', metavar='NUMBER',
            help='Pick images from trajectory.  NUMBER can be a '
            'single number (use a negative number to count from '
            'the back) or a range: start:stop:step, where the '
            '":step" part can be left out - default values are '
            '0:nimages:1.')
        add('--read-args', nargs='+', action='store',
            default={}, metavar="KEY=VALUE",
            help='Additional keyword arguments to pass to '
            '`ase.io.read()`.')

    @staticmethod
    def run(args, parser):
        if not (args.exec_code or args.exec_file):
            parser.error("Atleast one of '-e' or '-E' must be provided")

        if args.read_args:
            args.read_args = eval("dict({0})"
                                  .format(', '.join(args.read_args)))

        configs = []
        for filename in args.input:
            atoms = read(filename, args.image_number,
                         format=args.input_format, **args.read_args)
            if isinstance(atoms, list):
                configs.extend(atoms)
            else:
                configs.append(atoms)

        for atoms in configs:
            if args.exec_code:
                # avoid exec() for Py 2+3 compat.
                eval(compile(args.exec_code, '<string>', 'exec'))
            if args.exec_file:
                eval(compile(open(args.exec_file).read(), args.exec_file,
                             'exec'))
