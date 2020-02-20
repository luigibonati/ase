import string
from ase.io import string2index
from ase.io.formats import parse_filename
from ase.data import chemical_symbols
import numpy as np

# default fields


def field_specs_on_conditions(calculator_outputs, rank_order):
    if calculator_outputs:
        field_specs = ['i:0', 'el', 'd', 'rd', 'df', 'rdf']
    else:
        field_specs = ['i:0', 'el', 'dx', 'dy', 'dz', 'd', 'rd']
    if rank_order is not None:
        if rank_order in field_specs:
            for c, i in enumerate(field_specs):
                if i == rank_order:
                    field_specs[c] = i + ':0:1'
        else:
            field_specs.append(rank_order + ':0:1')
    else:
        field_specs[0] = field_specs[0] + ':1'
    return field_specs


def summary_functions_on_conditions(has_calc):
    if has_calc:
        return [rmsd, energy_delta]
    return [rmsd]


def header_alias(h):
    """Replace keyboard characters with Unicode symbols for pretty printing"""
    if h == 'i':
        h = 'index'
    elif h == 'an':
        h = 'atomic #'
    elif h == 't':
        h = 'tag'
    elif h == 'el':
        h = 'element'
    elif h[0] == 'd':
        h = h.replace('d', 'Δ')
    elif h[0] == 'r':
        h = 'rank ' + header_alias(h[1:])
    elif h[0] == 'a':
        h = h.replace('a', '<')
        h += '>'
    return h


def prec_round(a, prec=2):
    "To make hierarchical sorting different from non-hierarchical sorting with floats"
    if a == 0:
        return a
    else:
        s = 1 if a > 0 else -1
        m = np.log(s * a) // 1
        c = np.log(s * a) % 1
    return s * np.round(np.exp(c), prec) * np.exp(m)


prec_round = np.vectorize(prec_round)

# end most settings


def sort2rank(sort):
    """
    Given an argsort, return a list which gives the rank of the element at each position.
    Also does the inverse problem (an involutive transform) of given a list of ranks of the elements, return an argsort.
    """
    n = len(sort)
    rank = np.zeros(n, dtype=int)
    for i in range(n):
        rank[sort[i]] = i
    return rank


num2sym = dict(zip(np.argsort(chemical_symbols), chemical_symbols))
sym2num = {v: k for k, v in num2sym.items()}


def get_data(atoms1, atoms2, field):
    if field[0] == 'r':
        field = field[1:]
        rank_order = True
    else:
        rank_order = False

    if 'd' == field[0]:
        dpositions = atoms2.positions - atoms1.positions

    if field == 'dx':
        data = dpositions[:, 0]
    elif field == 'dy':
        data = dpositions[:, 1]
    elif field == 'dz':
        data = dpositions[:, 2]
    elif field == 'd':
        data = np.linalg.norm(dpositions, axis=1)
    elif field == 't':
        data = atoms1.get_tags()
    elif field == 'an':
        data = atoms1.numbers
    if field == 'el':
        data = np.array([sym2num[sym] for sym in atoms1.symbols])
    elif field == 'i':
        data = np.arange(len(atoms1))
    if rank_order:
        return sort2rank(np.argsort(-data))

    return data


atoms_props = ['dx', 'dy', 'dz', 'd', 't', 'an', 'i', 'el']


def get_atoms_data(atoms1, atoms2, field):

    if field.lstrip('r') in atoms_props:
        return get_data(atoms1, atoms2, field)

    if field[0] == 'r':
        field = field[1:]
        rank_order = True
    else:
        rank_order = False

    if 'd' == field[0]:
        dforces = atoms2.get_forces() - atoms1.get_forces()
    else:
        aforces = (atoms2.get_forces() + atoms1.get_forces()) / 2

    if field == 'dfx':
        data = dforces[:, 0]
    elif field == 'dfy':
        data = dforces[:, 1]
    elif field == 'dfz':
        data = dforces[:, 2]
    elif field == 'df':
        data = np.linalg.norm(dforces, axis=1)
    elif field == 'afx':
        data = aforces[:, 0]
    elif field == 'afy':
        data = aforces[:, 1]
    elif field == 'afz':
        data = aforces[:, 2]
    elif field == 'af':
        data = np.linalg.norm(aforces, axis=1)

    if rank_order:
        return sort2rank(np.argsort(-data))

    return data


# Summary Functions

def rmsd(atoms1, atoms2):
    dpositions = atoms2.positions - atoms1.positions
    rmsd = ((np.linalg.norm(dpositions, axis=1)**2).mean())**(0.5)
    return 'RMSD={:+.1E}'.format(rmsd)


def energy_delta(atoms1, atoms2):
    E1 = atoms1.get_potential_energy()
    E2 = atoms2.get_potential_energy()
    return 'E1 = {:+.1E}, E2 = {:+.1E}, dE = {:+1.1E}'.format(E1, E2, E2 - E1)


def parse_field_specs(field_specs):
    fields = []
    hier = []
    scent = []
    for fs in field_specs:
        hsf = fs.split(':')
        if len(hsf) == 3:
            scent.append(int(hsf[2]))
            hier.append(int(hsf[1]))
            fields.append(hsf[0])
        elif len(hsf) == 2:
            scent.append(-1)
            hier.append(int(hsf[1]))
            fields.append(hsf[0])
        elif len(hsf) == 1:
            scent.append(-1)
            hier.append(-1)
            fields.append(hsf[0])

    hier = np.array(hier)
    mxm = max(hier)
    for c in range(len(hier)):
        if hier[c] < 0:
            mxm += 1
            hier[c] = mxm
    # reversed by convention of numpy lexsort
    hier = sort2rank(np.array(hier))[::-1]
    return fields, hier, scent

# Class definitions


class DiffTemplate(string.Formatter):
    """Changing string formatting method to convert numeric data field"""

    # map set of strings to integers for numeric sorting and reverse map

    def format_field(self, value, spec):
        if spec.endswith('h'):
            value = num2sym[int(value)]  # cast to int since it will be float
            spec = spec[:-1] + 's'
        return super(DiffTemplate, self).format_field(value, spec)


class Table(object):
    def __init__(self,
                 field_specs,
                 summary_functions=[],
                 max_lines=None,
                 title='',
                 toprule='=',
                 bottomrule='=',
                 midrule='-',
                 tablewidth=None,
                 columnwidth=9,
                 precision=2,
                 representation='E'):

        self.max_lines = max_lines
        self.summary_functions = summary_functions
        self.field_specs = field_specs

        self.fields, self.hier, self.scent = parse_field_specs(self.field_specs)
        self.nfields = len(self.fields)

        # formatting
        self.precision = precision
        self.representation = representation
        self.columnwidth = columnwidth
        if tablewidth is None:
            self.tablewidth = columnwidth * self.nfields
        else:
            self.tablewidth = tablewidth
        self.formatter = DiffTemplate().format
        self.fmt_class = {
            'signed float': "{{: ^{}.{}{}}}".format(
                self.columnwidth,
                self.precision - 1,
                self.representation),
            'unsigned float': "{{:^{}.{}{}}}".format(
                self.columnwidth,
                self.precision - 1,
                self.representation),
            'int': "{{:^{}n}}".format(
                self.columnwidth),
            'str': "{{:^{}s}}".format(
                self.columnwidth),
            'conv': "{{:^{}h}}".format(
                self.columnwidth)}

        self.fmt = self.make_fmt()
        self.title = title
        self.header = self.make_header()
        self.toprule = toprule * self.tablewidth
        self.bottomrule = bottomrule * self.tablewidth
        self.midrule = midrule * self.tablewidth

    def make_fmt(self):
        fmt = {}
        signed_floats = [
            'dx',
            'dy',
            'dz',
            'dfx',
            'dfy',
            'dfz',
            'afx',
            'afy',
            'afz']
        for sf in signed_floats:
            fmt[sf] = self.fmt_class['signed float']
        unsigned_floats = ['d', 'df', 'af']
        for usf in unsigned_floats:
            fmt[usf] = self.fmt_class['unsigned float']
        integers = ['i', 'an', 't'] + ['r' + sf for sf in signed_floats] + \
            ['r' + usf for usf in unsigned_floats]
        for i in integers:
            fmt[i] = self.fmt_class['int']
        fmt['el'] = self.fmt_class['conv']
        return fmt

    def make(self, atoms1, atoms2):
        body = self.make_body(atoms1, atoms2)
        if self.max_lines is not None:
            body = body[:self.max_lines]
        summary = self.make_summary(atoms1, atoms2)

        return '\n'.join([self.title, self.toprule, self.header,
                          self.midrule, body, self.bottomrule, summary])

    def make_header(self):
        return self.formatter(
            self.fmt_class['str'] * self.nfields, *[header_alias(field) for field in self.fields])

    def make_summary(self, atoms1, atoms2):
        return '\n'.join([summary_function(atoms1, atoms2)
                          for summary_function in self.summary_functions])

    def make_body(self, atoms1, atoms2):
        fdata = np.array([get_atoms_data(atoms1, atoms2, field)
                          for field in self.fields])
        sorting_array = prec_round(
            (np.array(self.scent)[:, np.newaxis] * fdata)[self.hier])
        data = fdata[:, np.lexsort(sorting_array)].transpose()
        rowformat = ''.join([self.fmt[field] for field in self.fields])
        body = [self.formatter(rowformat, *row) for row in data]
        return '\n'.join(body)


default_index = string2index(':')


def slice_split(filename):
    if '@' in filename:
        filename, index = parse_filename(filename, None)
    else:
        filename, index = parse_filename(filename, default_index)
    return filename, index
