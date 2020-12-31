import os

from ase.io import read, write


class CLICommand:
    """ Execute code on files.

    The given python code is evaluated on the atoms object where 'atoms' is
    used to denote the Atoms object.
    """

    @staticmethod
    def add_arguments(parser):
        add = parser.add_argument
        add('input', nargs='+', metavar='input-file')
        add('-i', '--input-format', metavar='FORMAT',
            help='Specify input FORMAT')
        add('-n', '--image-number',
            default=':', metavar='NUMBER',
            help='Pick images from trajectory.  NUMBER can be a '
            'single number (use a negative number to count from '
            'the back) or a range: start:stop:step, where the '
            '":step" part can be left out - default values are '
            '0:nimages:1.')
        add('-e', '--exec-code',
            help='Python code to execute on each atoms before '
            'writing it to output file. The Atoms object is '
            'available as `atoms`. Set `atoms.info["_output"] = False` '
            'to suppress output of this frame.')
        add('-E', '--exec-file',
            help='Python source code file to execute on each '
            'frame, usage is as for -e/--exec-code.')
        add('--read-args', nargs='+', action='store',
            default={}, metavar="KEY=VALUE",
            help='Additional keyword arguments to pass to '
            '`ase.io.read()`.')
        add('--write-args', nargs='+', action='store',
            default={}, metavar="KEY=VALUE",
            help='Additional keyword arguments to pass to '
            '`ase.io.write()`.')

    @staticmethod
    def run(args, parser):
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
