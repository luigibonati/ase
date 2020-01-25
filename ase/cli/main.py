import argparse
import sys
import textwrap
import re

from ase import __version__
from ase.utils import import_module


class CLIError(Exception):
    """Error for CLI commands.

    A subcommand may raise this.  The message will be forwarded to
    the error() method of the argument parser."""


# Important: Following any change to command-line parameters, use
# python3 -m ase.cli.completion to update autocompletion.
commands = [
    ('info', 'ase.cli.info'),
    # ('show', 'ase.cli.show'),
    ('test', 'ase.test'),
    ('gui', 'ase.gui.ag'),
    ('db', 'ase.db.cli'),
    ('run', 'ase.cli.run'),
    ('band-structure', 'ase.cli.band_structure'),
    ('build', 'ase.cli.build'),
    ('eos', 'ase.eos'),
    ('ulm', 'ase.io.ulm'),
    ('find', 'ase.cli.find'),
    ('nomad-upload', 'ase.cli.nomad'),
    ('nomad-get', 'ase.cli.nomadget'),
    ('convert', 'ase.cli.convert'),
    ('reciprocal', 'ase.cli.reciprocal'),
    ('completion', 'ase.cli.completion'),
    ('diff', 'ase.cli.diff')
]


def main(prog='ase', description='ASE command line tool.',
         version=__version__, commands=commands, hook=None, args=None):
    parser = argparse.ArgumentParser(prog=prog,
                                     description=description,
                                     formatter_class=Formatter)
    parser.add_argument('--version', action='version',
                        version='%(prog)s-{}'.format(version))
    parser.add_argument('-T', '--traceback', action='store_true')
    subparsers = parser.add_subparsers(title='Sub-commands',
                                       dest='command')

    subparser = subparsers.add_parser('help',
                                      description='Help',
                                      help='Help for sub-command.')
    subparser.add_argument('helpcommand',
                           nargs='?',
                           metavar='sub-command',
                           help='Provide help for sub-command.')

    functions = {}
    parsers = {}
    for command, module_name in commands:
        cmd = import_module(module_name).CLICommand
        docstring = cmd.__doc__
        if docstring is None:
            # Backwards compatibility with GPAW
            short = cmd.short_description
            long = getattr(cmd, 'description', short)
        else:
            parts = docstring.split('\n', 1)
            if len(parts) == 1:
                short = docstring
                long = docstring
            else:
                short, body = parts
                long = short + '\n' + textwrap.dedent(body)
        subparser = subparsers.add_parser(
            command,
            formatter_class=Formatter,
            help=short,
            description=long)
        cmd.add_arguments(subparser)
        functions[command] = cmd.run
        parsers[command] = subparser

    if hook:
        args = hook(parser, args)
    else:
        args = parser.parse_args(args)

    if args.command == 'help':
        if args.helpcommand is None:
            parser.print_help()
        else:
            parsers[args.helpcommand].print_help()
    elif args.command is None:
        parser.print_usage()
    else:
        f = functions[args.command]
        try:
            if f.__code__.co_argcount == 1:
                f(args)
            else:
                f(args, parsers[args.command])
        except KeyboardInterrupt:
            pass
        except CLIError as x:
            parser.error(x)
        except Exception as x:
            if args.traceback:
                raise
            else:
                l1 = '{}: {}\n'.format(x.__class__.__name__, x)
                l2 = ('To get a full traceback, use: {} -T {} ...'
                      .format(prog, args.command))
                parser.error(l1 + l2)


class Formatter(argparse.HelpFormatter):
    """Improved help formatter."""

    def _fill_text(self, text, width, indent):
        assert indent == ''
        out = ''
        blocks = text.split('\n\n')
        for block in blocks:
            if block[0] == '*':
                # List items:
                for item in block[2:].split('\n* '):
                    out += textwrap.fill(item,
                                         width=width - 2,
                                         initial_indent='* ',
                                         subsequent_indent='  ') + '\n'
            elif block[0] == ' ':
                # Indented literal block:
                out += block + '\n'
            else:
                # Block of text:
                out += textwrap.fill(block, width=width) + '\n'
            out += '\n'
        return out[:-1]

    def _extended_help_formatter(self, help_text, width, indent):
        """Format the help to fit in columns when the help is more complicated
    (is long and contains lists) under the constraint that items are not
    broken between lines."""

        # Since the source code also has to follow some
        # indenting conventions first the input strings must have all whitespace
        # be removed.


        help_text = re.sub(r'[ \t]+', ' ', help_text)
        blocks = [[line for line in block.split('\n')] for block in help_text.split(
            '\n\n')]  # how to deal with mid-sentence truncations?

        # then concatenate strings which don't start with '* ', the item
        # designator, to be formatted together

        new_blocks = []
        for block in blocks:
            new_block = []
            carry_string = ''
            for line in block:
                if line.lstrip()[:2] == '* ':
                    if carry_string != '':
                        new_block.append(carry_string)
                        carry_string = ''
                    new_block.append(line)
                else:
                    carry_string += line
            new_block.append(carry_string)
            new_blocks.append(new_block)

        new_blocks = [[line for line in block if line != '']
                      for block in new_blocks]  # clean up if logic is bad

        # use an external text wrapper program

        out = ''
        for block in new_blocks:
            for line in block:
                out += '\n' + textwrap.fill(line, width=width)
            out += '\n'

        # strip white space and left justify
        indent = ' ' * indent
        s = ('\n' + indent).join([i.strip() for i in out.split('\n')])
        s = s.strip('\n').rstrip(' ')

        return s

    def _format_action(self, action):
        # determine the required width and the entry label
        help_position = min(self._action_max_length + 2,
                            self._max_help_position)
        help_width = max(self._width - help_position, 11)
        action_width = help_position - self._current_indent - 2
        action_header = self._format_action_invocation(action)

        # no help; start on same line and add a final newline
        if not action.help:
            tup = self._current_indent, '', action_header
            action_header = '%*s%s\n' % tup

        # short action name; start on the same line and pad two spaces
        elif len(action_header) <= action_width:
            tup = self._current_indent, '', action_width, action_header
            action_header = '%*s%-*s  ' % tup
            indent_first = 0

        # long action name; start on the next line
        else:
            tup = self._current_indent, '', action_header
            action_header = '%*s%s\n' % tup
            indent_first = help_position

        # collect the pieces of the action help
        parts = [action_header]

        # if there was help for the action, add lines of help text
        if action.help:
            help_text = self._expand_help(action)
            s = self._extended_help_formatter(
                help_text, help_width, help_position)
            if indent_first == 0:
                s = s.lstrip()
            parts.append(s)

        # or add a newline if the description doesn't end with one
        elif not action_header.endswith('\n'):
            parts.append('\n')

        # if there are any sub-actions, add their help as well
        for subaction in self._iter_indented_subactions(action):
            parts.append(self._format_action(subaction))

        # return a single string
        return self._join_parts(parts)


def old():
    cmd = sys.argv[0].split('-')[-1]
    print('Please use "ase {cmd}" instead of "ase-{cmd}"'.format(cmd=cmd))
    sys.argv[:1] = ['ase', cmd]
    main()
