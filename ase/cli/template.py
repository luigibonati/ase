import numpy as np

template = "{title}\n{toprule}\n{header}\n{midrule}\n{body}\n{bottomrule}\n{summary}"
fd={}
cwidth = 72
fd['toprule']=fd['bottomrule']='='*cwidth
fd['midrule']='-'*cwidth
fd['title'] = 'Coordinates Summary'
fd2 = fd.copy()
fd2['title'] = 'Forces and Coordinates Summary'

tw=str(9)
pre='{:^'+tw
fmt1 = {'signed float':pre[:-1]+' '+pre[-1]+'.1E}',
        'unsigned float':pre+'.1E}',
        'int':pre+'n}',
        'str':pre+'s}',
        'conv':pre+'h}'}

fmt={}

l1 = ['dx','dy','dz','dfx','dfy','dfz','afx','afy','afz']

def header_rules(i):
    if i == 'i':
        i = 'index'
    elif i == 'an':
        i = 'atomic #'
    elif i == 't':
        i = 'tag'
    elif i[0] == 'd':
        i = i.replace('d','Δ')
    elif i[0] == 'r':
        i = i.replace('r','rank ')
    elif i[0] == 'a':
        i = i.replace('a','<')
        i+='>'
    return i

for i in l1:
    fmt[i] = 'signed float'
l2 = ['d','df','af']
for i in l2:
    fmt[i] = 'unsigned float'
l3 = ['i','an','t'] + ['r'+i for i in l1] + ['r' + i for i in l2]
for i in l3:
    fmt[i] = 'int'
fmt['el'] = 'conv'

fields = ['i:0','el','dx','dy','dz','d','rd']
fields_calculator_outputs = ['i:0','el','d','rd','df','rdf']

