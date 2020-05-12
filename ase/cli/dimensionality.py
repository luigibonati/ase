from ase.io import read
from ase.geometry.dimensionality import analyze_dimensionality
import warnings


class CLICommand:
    """Analyze dimensionality of a structure. Takes any file format
    supported by ASE.

    ase dimensionality structure.cif
    """

    @staticmethod
    def add_arguments(parser):
        add = parser.add_argument
        add('filename', help='input file to analyze')

    @staticmethod
    def run(args, parser):
        atoms = read(args.filename)
        result = analyze_dimensionality(atoms)

        print('type   score     a      b      component counts')
        print('===============================================')
        for entry in result:
            dimtype = entry.dimtype.rjust(4)
            score = '{:.3f}'.format(entry.score).ljust(5)
            a = '{:.3f}'.format(entry.a).ljust(5)
            b = '{:.3f}'.format(entry.b).ljust(5)
            line = '{}   {}   {}   {}   {}'.format(dimtype, score, a, b,
                                                   entry.h)
            print(line)


# reading CIF files can produce a ton of distracting warnings
warnings.filterwarnings("ignore")
