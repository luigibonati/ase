import numpy as np
#settings
template = "{title}\n{toprule}\n{header}\n{midrule}\n{body}\n{bottomrule}\n{summary}"
twidth = 72 
tw=str(9)
field_specs = ['i:0:1','el','dx','dy','dz','d','rd']
field_specs_calc = ['i:0:1','el','d','rd','df','rdf']

#template formatting dictionary (for coordinates only)
format_dict={}
format_dict['toprule']=format_dict['bottomrule']='='*twidth
format_dict['midrule']='-'*twidth
format_dict['title'] = 'Coordinates Summary'

format_dict_calc = format_dict.copy()
format_dict_calc['title'] = 'Forces and Coordinates Summary'

pre='{:^'+tw

fmt = {}
fmt_class = {'signed float':pre[:-1]+' '+pre[-1]+'.1E}',
        'unsigned float':pre+'.1E}',
        'int':pre+'n}',
        'str':pre+'s}',
        'conv':pre+'h}'}

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
        h = h.replace('d','Δ')
    elif h[0] == 'r':
        h = 'rank ' + header_alias(h[1:])
    elif h[0] == 'a':
        h = h.replace('a','<')
        h+='>'
    return h

l1 = ['dx','dy','dz','dfx','dfy','dfz','afx','afy','afz']
for i in l1:
    fmt[i] = fmt_class['signed float']
l2 = ['d','df','af']
for i in l2:
    fmt[i] = fmt_class['unsigned float']
l3 = ['i','an','t'] + ['r'+i for i in l1] + ['r' + i for i in l2]
for i in l3:
    fmt[i] = fmt_class['int']

fmt['el'] = fmt_class['conv']

def prec_round(a,prec=2):
    "To make hierarchical sorting different from non-hierarchical sorting with floats"
    if a == 0:
        return a
    else:
        s = 1 if a > 0 else -1
        m = np.log(s*a) // 1
        c = np.log(s*a) % 1
    return s*np.round(np.exp(c),prec) * np.exp(m)

prec_round = np.vectorize(prec_round)

import string
class DiffTemplate(string.Formatter):
    """Changing string formatting method to convert numeric data field"""
    def format_field(self, value, spec):
        if spec.endswith('h'):
            value = dct1[int(value)] # cast to int since it will be float
            spec = spec[:-1] + 's'
        return super(DiffTemplate, self).format_field(value, spec)

formatter = DiffTemplate().format

for field_spec in field_specs:
    if 'f' in field_spec:
        raise Exception('setting of a calculator output in coordinate fields')
