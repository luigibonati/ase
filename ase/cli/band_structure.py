import json
import numpy as np

from ase.io.jsonio import read_json
from ase.spectrum.band_structure import BandStructure


class CLICommand:
    """Plot band-structure.

    Read eigenvalues and k-points from file and plot result

    Example:

        $ ase band-structure mybandstructure.json
    """

    @staticmethod
    def add_arguments(parser):
        parser.add_argument('file',
                            help='Band structure file')
        parser.add_argument('-o', '--output', help='Write image to a file')
        parser.add_argument('-r', '--range', nargs=2, default=['-3', '3'],
                            metavar=('emin', 'emax'),
                            help='Default: "-3.0 3.0" '
                            '(in eV relative to Fermi level)')

    @staticmethod
    def run(args, parser):
        main(args, parser)


def read_band_structure(args, parser):
    try:
        bs = read_json(args.file)
    except json.decoder.JSONDecodeError:
        parser.error('File resembles neither atoms nor band structure')

    objtype = getattr(bs, 'ase_objtype', None)
    if objtype != 'bandstructure':
        parser.error('Expected band structure, but this file contains a {}'
                     .format(objtype or type(bs)))

    return bs


def main(args, parser):
    import matplotlib.pyplot as plt
    bs = read_band_structure(args, parser)
    emin, emax = (float(e) for e in args.range)
    fig = plt.gcf()
    fig.canvas.set_window_title(args.file)
    ax = fig.gca()
    bs.plot(ax=ax,
            filename=args.output,
            emin=emin + bs.reference,
            emax=emax + bs.reference)
    if args.output is None:
        plt.show()
