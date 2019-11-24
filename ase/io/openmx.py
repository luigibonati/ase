from ase.calculators.openmx.writer import (parameters_to_keywords,
                                           get_standard_key)


def write_openmx_in(fd, atoms, properties=None, label='openmx',
                    parameters=None):
    if parameters is None:
        parameters = {}
    from ase.calculators.openmx import parameters as param
    filtered_keywords = parameters_to_keywords(label=label, atoms=atoms,
                                               parameters=parameters,
                                               properties=properties)
    keys = ['string', 'bool', 'integer', 'float',
            'tuple_integer', 'tuple_float', 'tuple_bool',
            'matrix', 'list_int', 'list_bool', 'list_float']
    # Write 1-line keywords
    for fltrd_keyword in filtered_keywords.keys():
        for key in keys:
            openmx_keywords = getattr(param, key+'_keys')
            write = globals()['write_'+key]
            for omx_keyword in openmx_keywords:
                if fltrd_keyword == get_standard_key(omx_keyword):
                    write(fd, omx_keyword, filtered_keywords[fltrd_keyword])


def write_string(f, key, value):
    f.write("        ".join([key, value]))
    f.write("\n")


def write_tuple_integer(f, key, value):
    f.write("        ".join([key, "%d %d %d" % value]))
    f.write("\n")


def write_tuple_float(f, key, value):
    f.write("        ".join([key, "%.4f %.4f %.4f" % value]))
    f.write("\n")


def write_tuple_bool(f, key, value):
    omx_bl = {True: 'On', False: 'Off'}
    f.write("        ".join([key, "%s %s %s" % [omx_bl[bl] for bl in value]]))
    f.write("\n")


def write_integer(f, key, value):
    f.write("        ".join([key, "%d" % value]))
    f.write("\n")


def write_float(f, key, value):
    f.write("        ".join([key, "%.8g" % value]))
    f.write("\n")


def write_bool(f, key, value):
    omx_bl = {True: 'On', False: 'Off'}
    f.write("        ".join([key, "%s" % omx_bl[value]]))
    f.write("\n")


def write_list_int(f, key, value):
    f.write("".join(key) + "     ".join(map(str, value)))


def write_list_bool(f, key, value):
    omx_bl = {True: 'On', False: 'Off'}
    f.write("".join(key) + "     ".join([omx_bl[bl] for bl in value]))


def write_list_float(f, key, value):
    f.write("".join(key) + "     ".join(map(str, value)))


def write_matrix(f, key, value):
    f.write('<' + key)
    f.write("\n")
    for line in value:
        f.write("    "+"  ".join(map(str, line)))
        f.write("\n")
    f.write(key + '>')
    f.write("\n\n")
