def remove_doubles(string):
    string2 = string.replace(' ' * 2, ' ')
    if string2 == string:
        return string2
    else:
        return remove_doubles(string2)

def extended_help_formatter(help_text, width, indent):
    """Format the help to fit in columns when the help is more complicated
(is long and contains lists) under the constraint that items are not
broken between lines."""

    # Since the source code also has to follow some
    # indenting conventions first the input strings must have all whitespace
    # be removed.

    help_text = remove_doubles(help_text)
    blocks = [[line for line in block.split('\n')]
              for block in help_text.split('\n\n')] # how to deal with mid-sentence truncations?

    # then concatenate strings which don't start with '* ', the item
    # designator, to be formatted together

    new_blocks = []
    for block in blocks:
        new_block = []
        carry_string = ''
        for line in block:
            if line[:3] == ' * ':
                if carry_string != '':
                    new_block.append(carry_string)
                    carry_string = ''
                new_block.append(line)
            else:
                carry_string += line
        new_block.append(carry_string)
        new_blocks.append(new_block)

    new_blocks = [[line for line in block if line != '']
                  for block in new_blocks]  # clean up if logic is bad

    # use an external text wrapper program
    import textwrap

    def tw(item):
        return textwrap.fill(item, width=width)

    out = ''
    for block in new_blocks:
        for line in block:
            out += '\n' + tw(line)
        out += '\n'

    # strip white space and left justify 
    i = ' ' * indent
    s = ('\n' + i).join([i.strip() for i in out.split('\n')])
    s = s.strip('\n').rstrip(' ')

    return s


if __name__ == '__main__':
    help_text = """Without argument,looks for ~/.ase/template.py.  Otherwise,
                        expects the comma separated list of the fields to include
                        in their left-to-right order.  Optionally, specify the
                        lexicographical sort hierarchy (0 is outermost sort) and if the
                        sort should be ascending or descending (1 or -1).  By default,
                        sorting is descending, which makes sense for most things except
                        index (and rank, but one can just sort by the thing which is
                        ranked to get ascending ranks).

                        * example: ase diff start.cif stop.cif --template
                        * i:0:1,el,dx,dy,dz,d,rd

                        possible fields:

                        * i: index
                        * dx,dy,dz,d: displacement/displacement components
                        * dfx,dfy,dfz,df: difference force/force components
                        * afx,afy,afz,af: average force/force components
                        * an: atomic number
                        * el: atomic element
                        * t: atom tag
                        * r<col>: the rank of that atom with respect to the column
            """
    t = extended_help_formatter(help_text, 20, 5)
    print(t)
