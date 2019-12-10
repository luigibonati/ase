import string
import numpy as np
# settings
template = "{title}\n{toprule}\n{header}\n{midrule}\n{body}\n{bottomrule}\n{summary}"
twidth = 72
tw = str(9)
def field_specs_on_conditions(calculator_outputs,rank_order):
    if calculator_outputs:
        field_specs = ['i:0', 'el', 'd', 'rd', 'df', 'rdf']
    else:
        field_specs = ['i:0', 'el', 'dx', 'dy', 'dz', 'd', 'rd']
    if rank_order != None:
        if rank_order in field_specs:
            for c,i in enumerate(field_specs):
                if i == rank_order:
                    field_specs[c] = i + ':0:1'
        else:
            field_specs.append(rank_order+':0:1')
    else:
        field_specs[0] = field_specs[0]+':1'
    return field_specs

# template formatting dictionary (for coordinates only)
format_dict = {}
format_dict['toprule'] = format_dict['bottomrule'] = '=' * twidth
format_dict['midrule'] = '-' * twidth
format_dict['title'] = 'Coordinates Summary'

format_dict_calc = format_dict.copy()
format_dict_calc['title'] = 'Forces and Coordinates Summary'

pre = '{:^' + tw

fmt = {}
fmt_class = {'signed float': pre[:-1] + ' ' + pre[-1] + '.1E}',
             'unsigned float': pre + '.1E}',
             'int': pre + 'n}',
             'str': pre + 's}',
             'conv': pre + 'h}'}


def header_alias(h):
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


l1 = ['dx', 'dy', 'dz', 'dfx', 'dfy', 'dfz', 'afx', 'afy', 'afz']
for i in l1:
    fmt[i] = fmt_class['signed float']
l2 = ['d', 'df', 'af']
for i in l2:
    fmt[i] = fmt_class['unsigned float']
l3 = ['i', 'an', 't'] + ['r' + i for i in l1] + ['r' + i for i in l2]
for i in l3:
    fmt[i] = fmt_class['int']

fmt['el'] = fmt_class['conv']


def prec_round_1(a, prec=2):
    "To make hierarchical sorting different from non-hierarchical sorting with floats"
    if a == 0:
        return a
    else:
        s = 1 if a > 0 else -1
        m = np.log(s * a) // 1
        c = np.log(s * a) % 1
    return s * np.round(np.exp(c), prec) * np.exp(m)

prec_round = np.vectorize(prec_round_1)

# end most settings

from ase.data import chemical_symbols
num2sym = dict(zip(np.argsort(chemical_symbols),chemical_symbols))
sym2num = {v: k for k, v in num2sym.items()}

class DiffTemplate(string.Formatter):
    """Changing string formatting method to convert numeric data field"""

    def format_field(self, value, spec):
        if spec.endswith('h'):
            value = num2sym[int(value)]  # cast to int since it will be float
            spec = spec[:-1] + 's'
        return super(DiffTemplate, self).format_field(value, spec)

formatter = DiffTemplate().format

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


def get_data(atoms1, atoms2, field):
    if field[0] == 'r':
        field = field[1:]
        rank_order = True
    else:
        rank_order = False

    if field == 'dx':
        data = -atoms1.positions[:, 0] + atoms2.positions[:, 0]
    elif field == 'dy':
        data = -atoms1.positions[:, 1] + atoms2.positions[:, 1]
    elif field == 'dz':
        data = -atoms1.positions[:, 2] + atoms2.positions[:, 2]
    elif field == 'd':
        data = np.linalg.norm(atoms1.positions - atoms2.positions, axis=1)
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

def get_images_data(images1, images2, counter, field):
    if images1 == images2:
        atoms1 = images1.get_atoms(counter - 1)
        atoms2 = images2.get_atoms(counter)
    else:
        atoms1 = images1.get_atoms(counter)
        atoms2 = images2.get_atoms(counter)

    if field.lstrip('r') in atoms_props:
        return get_data(atoms1, atoms2, field)

    if field[0] == 'r':
        field = field[1:]
        rank_order = True
    else:
        rank_order = False

    if field == 'dfx':
        data = -images1.get_forces(atoms1)[:, 0] + \
            images2.get_forces(atoms2)[:, 0]
    elif field == 'dfy':
        data = -images1.get_forces(atoms1)[:, 1] + \
            images2.get_forces(atoms2)[:, 1]
    elif field == 'dfz':
        data = -images1.get_forces(atoms1)[:, 2] + \
            images2.get_forces(atoms2)[:, 2]
    elif field == 'df':
        data = np.linalg.norm(
            images1.get_forces(atoms1) -
            images2.get_forces(atoms2),
            axis=1)
    elif field == 'afx':
        data = images1.get_forces(
            atoms1)[:, 0] + images2.get_forces(atoms2)[:, 0]
        data /= 2
    elif field == 'afy':
        data = images1.get_forces(
            atoms1)[:, 1] + images2.get_forces(atoms2)[:, 1]
        data /= 2
    elif field == 'afz':
        data = images1.get_forces(
            atoms1)[:, 2] + images2.get_forces(atoms2)[:, 2]
        data /= 2
    elif field == 'af':
        data = np.linalg.norm(
            images1.get_forces(atoms1) +
            images2.get_forces(atoms2),
            axis=1)
        data /= 2

    if rank_order:
        return sort2rank(np.argsort(-data))
    return data


def format_table(table, fields):
    s = ''.join([fmt[field] for field in fields])
    body = [formatter(s, *row) for row in table]
    return '\n'.join(body)


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
    if hier.all() == -1:
        hier = []
    else:
        mxm = max(hier)
        for c in range(len(hier)):
            if hier[c] < 0:
                mxm += 1
                hier[c] = mxm
        # reversed by convention of numpy lexsort
        hier = sort2rank(np.array(hier))[::-1]
    return fields, hier, scent


def render_table(field_specs, atoms1, atoms2, show_only=None):
    fields, hier, scent = parse_field_specs(field_specs)
    fdata = np.array([get_data(atoms1, atoms2, field) for field in fields])
    if hier != []:
        sorting_array = prec_round(
            (np.array(scent)[:, np.newaxis] * fdata)[hier])
        table = fdata[:, np.lexsort(sorting_array)].transpose()
    else:
        table = fdata.transpose()
    if show_only is not None:
        table = table[:show_only]
    format_dict['body'] = format_table(table, fields)
    format_dict['header'] = (fmt_class['str'] *
                             len(fields)).format(*
                                                 [header_alias(field) for field in fields])
    format_dict['summary'] = 'RMSD={:+.1E}'.format(
        np.sqrt(
            np.power(
                np.linalg.norm(
                    atoms1.positions -
                    atoms2.positions,
                    axis=1),
                2).mean()))
    return template.format(**format_dict)


def render_table_calc(field_specs, images1, images2, counter, show_only=None):
    fields, hier, scent = parse_field_specs(field_specs)
    fdata = np.array([get_images_data(images1, images2, counter, field)
                      for field in fields])
    if hier != []:
        sorting_array = prec_round(
            (np.array(scent)[:, np.newaxis] * fdata)[hier])
        table = fdata[:, np.lexsort(sorting_array)].transpose()
    else:
        table = fdata.transpose()
    if show_only is not None:
        table = table[:show_only]
    format_dict_calc['body'] = format_table(table, fields)
    format_dict_calc['header'] = (fmt_class['str'] *
                                  len(fields)).format(*
                                                      [header_alias(field) for field in fields])
    if images1 == images2:
        E1 = images1.get_energy(images1.get_atoms(counter))
        E2 = images2.get_energy(images2.get_atoms(counter + 1))
    else:
        E1 = images1.get_energy(images1.get_atoms(counter))
        E2 = images2.get_energy(images2.get_atoms(counter))
    format_dict_calc['summary'] = 'E{}={:+.1E}, E{}={:+.1E}, dE={:+.1E}'.format(
        counter, E1, counter + 1, E2, E2 - E1)
    return template.format(**format_dict_calc)


