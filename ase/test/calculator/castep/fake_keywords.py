fake_keywords_types = ['Real', 'String', 'Defined', 'Integer Vector',
                       'Boolean (Logical)', 'Integer', 'Real Vector',
                       'Block', 'Physical']

fake_keywords_levels = ['Dummy', 'Intermediate', 'Expert', 'Basic']

fake_keywords_data = {}

for kwt in fake_keywords_types:
    kw = 'fake_{0}_kw'.format(kwt.lower().replace(' ', '_'))
    if 'Boolean' in kwt:
        kw = 'fake_boolean_kw'
    fake_keywords_data[kw] = {
        'docstring': 'A fake {0} keyword'.format(kwt),
        'option_type': kwt,
        'keyword': kw,
        'level': 'Dummy'
    }