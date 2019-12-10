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
