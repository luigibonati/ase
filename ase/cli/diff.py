import numpy as np
from ase.data import chemical_symbols
from ase.io import read, iread
from ase.gui.images import Images
dct1 = dict(zip(np.argsort(chemical_symbols),chemical_symbols))
dct2 = {v: k for k, v in dct1.items()}

def get_data(atoms1,atoms2,field):
    if field[0] == 'r':
        field = field[1:]
        rank_order = True
    else:
        rank_order = False

    if field == 'dx':
        data = atoms1.positions[:,0] - atoms2.positions[:,0]
    elif field == 'dy':
        data = atoms1.positions[:,1] - atoms2.positions[:,1]
    elif field == 'dz':
        data = atoms1.positions[:,2] - atoms2.positions[:,2]
    elif field == 'd':
        data = np.linalg.norm(atoms1.positions - atoms2.positions,axis=1)
    elif field == 't':
        data = atoms1.get_tags()
    elif field == 'an':
        data = atoms1.numbers
    if field == 'el':
        data = np.array([dct2[symb] for symb in atoms1.symbols])
    elif field == 'i':
        data = np.arange(len(atoms1))

    if rank_order:
        return np.argsort(data)
    else:
        return data
    
def get_images_data(images1,images2,counter,field):
    if images1 == images2:
        atoms1=images1.get_atoms(counter-1)
        atoms2=images2.get_atoms(counter)
    else:
        atoms1=images1.get_atoms(counter)
        atoms2=images2.get_atoms(counter)

    atoms_props=['dx','dy','dz','d','t','an','i','el']

    if field in (atoms_props + ['r'+p for p in atoms_props]):
        return get_data(atoms1,atoms2,field)

    if field[0] == 'r':
        field = field[1:]
        rank_order = True
    else:
        rank_order = False

    if field == 'dfx':
        data = images.get_forces(atoms1)[:,0] - images.get_forces(atoms2)[:,0]
    elif field == 'dfy':
        data = images.get_forces(atoms1)[:,1] - images.get_forces(atoms2)[:,1]
    elif field == 'dfz':
        data = images.get_forces(atoms1)[:,2] - images.get_forces(atoms2)[:,2]
    elif field == 'df':
        data = np.linalg.norm(images1.get_forces(atoms1)- images2.get_forces(atoms2),axis=1)
    elif field == 'afx':
        data = images.get_forces(atoms1)[:,0] + images.get_forces(atoms2)[:,0]
        data /= 2
    elif field == 'afy':
        data = images.get_forces(atoms1)[:,1] + images.get_forces(atoms2)[:,1]
        data /= 2
    elif field == 'afz':
        data = images.get_forces(atoms1)[:,2] + images.get_forces(atoms2)[:,2]
        data /= 2
    elif field == 'af':
        data = np.linalg.norm(images1.get_forces(atoms1) + images2.get_forces(atoms2),axis=1)
        data /=2

    if rank_order: 
        return np.argsort(data)
    else:
        return data

#using only numpy with float datatype for ease
import string
class Template(string.Formatter):
    def format_field(self, value, spec):
        if spec.endswith('h'):
            value = dct1[int(value)] # cast to int since it will be float
            spec = spec[:-1] + 's'
        return super(Template, self).format_field(value, spec)

def format_body(rows,hier,scent):
    #permute the keys for numpy's lexsort
    s = [0]*len(rows) 
    for c,i in enumerate(hier):
        s[-(i+1)] = rows[c]*scent[c]

    s = np.lexsort(s)
    rows = [row[s] for row in rows]
    return np.transpose(rows)

from template import format_dict, format_dict_calc, fmt, fmt_class, header_alias, template

def format_table(table,fields):
    s = ''.join([fmt[field] for field in fields])
    formatter = Template().format
    body = [ formatter(s,*row) for row in table]
    return '\n'.join(body)

def render_table(field_specs,atoms1,atoms2):
    hier = [int(spec.split(':')[1]) if len(spec.split(':')) > 1 else -1 for spec in field_specs]
    scent = [int(spec.split(':')[2]) if len(spec.split(':')) > 2 else 1 for spec in field_specs]
    mxm = max(hier)
    for c in range(len(hier)):
        if hier[c] < 0:
            mxm += 1
            hier[c] = mxm
    fields = [spec.split(':')[0] for spec in field_specs]
    data = [get_data(atoms1, atoms2, field) for field in fields]
    table = format_body(data, hier, scent)
    format_dict['body'] = format_table(table, fields)
    format_dict['header'] = (fmt_class['str']*len(fields)).format(*[header_alias(field) for field in fields])
    format_dict['summary'] = 'RMSD={:+.1E}'.format(np.sqrt(np.power(np.linalg.norm(atoms1.positions-atoms2.positions,axis=1),2).mean()))
    return template.format(**format_dict)

def render_table_calc(field_specs,images1,images2,counter):
    hier = [int(spec.split(':')[1]) if len(spec.split(':')) > 1 else -1 for spec in field_specs]
    scent = [int(spec.split(':')[2]) if len(spec.split(':')) > 2 else 1 for spec in field_specs]
    mxm = max(hier)
    for c in range(len(hier)):
        if hier[c] < 0:
            mxm += 1
            hier[c] = mxm
    fields = [spec.split(':')[0] for spec in field_specs]
    data = [get_images_data(images1, images2, counter, field) for field in fields]
    table = format_body(data, hier, scent)
    format_dict_calc['body'] = format_table(table, fields)
    format_dict_calc['header'] = (fmt_class['str']*len(fields)).format(*[header_alias(field) for field in fields])
    if images1 == images2:
        E1 = images1.get_energy(images1.get_atoms(counter))
        E2 = images2.get_energy(images2.get_atoms(counter+1))
    else:
        E1 = images1.get_energy(images1.get_atoms(counter))
        E2 = images2.get_energy(images2.get_atoms(counter))
    format_dict_calc['summary'] = 'E{}={:+.1E}, E{}={:+.1E}, dE={:+.1E}'.format(counter, E1, counter + 1, E2, E2-E1)
    return template.format(**format_dict_calc)

from argparse import RawTextHelpFormatter
class CLICommand:
    """Print differences between atoms/calculations.

    Supports taking differences between different calculation runs of the same system as well as neighboring geometric images for one calculation run of a system. For those things which more than the difference is valuable, such as the magnitude of force or energy, the average magnitude between two images or the value for both images is given. Given the difference and the average as x and y, the two values can be calculated as x + y/2 and x - y/2.

    It is possible to fully customize the order of the columns and the sorting of the rows by their column fields.
    """

    @staticmethod
    def add_arguments(parser):
        add = parser.add_argument
        parser.formatter_class=RawTextHelpFormatter
        add('files', metavar='file(s)',
            help="""
            2 non-trajectory files: difference between them
            1 trajectory file: difference between consecutive images
            2 trajectory files: difference between corresponding image numbers""",
            nargs='+')
        add('-r','--rank-order', metavar='field', nargs='?', const='d',
            type=str, help="""order atoms by rank, possible ranks are\n
            an: atomic number
            d: change in position
            df: change in force
            af: average force in two iterations
            The default value, when specified, is d. 
            When not specified, ordering is the same as that provided by the generator. 
            For hiearchical sorting, see template.
            """)#,formatter_class=RawTextHelpFormatter)
        add('-c', '--calculator-outputs', action="store_true",
            help="display calculator outputs of forces and energy")
        add('-s', '--show-only', metavar='n', type=int, help="show only so many lines (atoms) in each table, useful if rank ordering")
        add('-t', '--template', metavar='template',nargs='?', const='rc', 
        help="""
        Without argument, looks for ~/.aserc/template.py.
        Otherwise, expects the comma separated list of the columns to include in their left-to-right order.
        Optionally, specify the lexicographical sort hierarchy (0 is outermost sort).

        example: ase diff start.cif stop.cif --template i,el:0,dx,dy,dz,d,rd:1

        possible columns:
            i: index
            dx,dy,dz,d: displacement/displacement components
            dfx,dfy,dfz,df: difference force/force components
            afx,afy,afz,af: average force/force components (between first and second iteration)
            an: atomic number
            el: atomic element
            t: atom tag
            r<col>: the rank of that atom with respect to the column

        it is possible to change formatters in a template file.
        """,
        
        )
        add('-l', '--log-file', metavar='logfile', help="print table to file")

    @staticmethod
    def run(args, parser):
        import sys 
        #output
        if args.log_file == None:
            out=sys.stdout
        else:
            out=open(args.log_file,'w')

        #templating
        if args.template == None:
            from template import fields, fields_calculator_outputs
            if args.calculator_outputs == True:
                fields = fields_calculator_outputs
        elif args.template == 'rc':
            import os
            homedir = os.environ['HOME']
            sys.path.insert(0,homedir+'/.ase.rc')
            from templaterc import format_dict, format_dict_calc, fmt, \
                    fmt_class, header_alias, template, \
                    fields, fields_calculator_outputs
            #this has to be named differently because python does not redundantly load 
            if args.calculator_outputs == False:
                fields = fields
            else:
                fields = fields_calculator_outputs
        else:
            fields = args.template.split(',')
            #raise error if user specified fields which require calculation outputs but that was not in the command line option
            #or just overwrite it

        if len(args.files) == 2:
            f1 = args.files[0]
            f2 = args.files[1]

            images1 = Images()
            images1.read([f1])
            images2 = Images()
            images2.read([f2])
            if 1.e-14 < images1.get_energy(images1.get_atoms(0)) < 1.e14:
                #this is a nan
                atoms1 = read(f1)
                atoms2 = read(f2)
                t = render_table(fields,atoms1,atoms2)
                print(t,file=out)
            else:
                if args.calculator_outputs == True:
                    for counter in range(len(images1)):
                        t = render_table_calc(fields,images1,images2,counter)
                        print(t,file=out)
                else:
                    for counter in range(len(images1)):
                        t = render_table(fields,images1.get_atoms(counter),images2.get_atoms(counter))
                        print(t,file=out)

        elif len(args.files) == 1:
            f1 = args.files[0]

            if args.calculator_outputs:
                images = Images()
                images.read([f1])
                for counter in range(len(images) - 1):
                    t = render_table_calc(fields,images,images,counter)
                    print(t,file=out)
            else:
                traj = iread(f1)
                atoms_list = list(traj)
                for counter in range(len(atoms_list)-1):
                    t = render_table(fields,atoms_list[counter],atoms_list[counter+1])
                    print(t,file=out)


if __name__ == "__main__":
    from ase.build import molecule
    atoms1 = molecule('CH4')
    atoms2 = molecule('CH4')
    atoms2.positions += np.random.random((5,3))/10.
    fields = ['dx','dy','dz','rd:1','i','an','t','el:0:1']
    images=Images()
    images.read(['vasprun.xml'])

    fields = ['dx','dy','dz','dfx','dfy','dfz','af', 'el:0:-1']
    s = [int(field.split(':')[1]) if len(field.split(':')) > 1 else -1 for field in fields]
    scent = [int(field.split(':')[2]) if len(field.split(':')) > 2 else 1 for field in fields]
    rem = len(s) - sum([1 if i != -1 else 0 for i in s]) 
    for c in range(len(s)):
        if s[c] == -1:
            s[c] = rem
            rem -= 1
    fields = [field.split(':')[0] for field in fields]
    data = [get_images_data(images, 1, field) for field in fields]
#    data = [get_data(atoms1, atoms2, field) for field in fields]
    mat = body(data, s, scent)
    body = format_table(mat, fields)
    header = (fmt1['str']*len(fields)).format(*[header_rules(field) for field in fields])
    print(header)
    [print(row) for row in body]
