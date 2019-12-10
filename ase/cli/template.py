import numpy as np
#settings
template = "{title}\n{toprule}\n{header}\n{midrule}\n{body}\n{bottomrule}\n{summary}"
twidth = 72 
tw=str(9)
fields = ['i:0','el','dx','dy','dz','d','rd']
fields_calculator_outputs = ['i:0','el','d','rd','df','rdf']

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
    elif h[0] == 'd':
        h = h.replace('d','Δ')
    elif h[0] == 'r':
        h = h.replace('r','rank ')
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


for field in fields:
    if 'f' in field:
        raise Exception('setting of a calculator output in coordinate fields')
