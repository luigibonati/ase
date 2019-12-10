import numpy as np
from ase.io import read, iread
from ase.gui.images import Images
from ase.cli.template import format_dict, format_dict_calc, fmt, fmt_class, header_alias, template, prec_round, formatter, num2sym, sym2num

def sort2rank(sort):
    """
    Given an argsort, return a list which gives the rank of the element at each position.
    Also does the inverse problem (an involutive transform) of given a list of ranks of the elements, return an argsort.
    """
    n = len(sort)
    rank = np.zeros(n,dtype=int)
    for i in range(n):
        rank[sort[i]] = i
    return rank

def get_data(atoms1,atoms2,field):
    if field[0] == 'r':
        field = field[1:]
        rank_order = True
    else:
        rank_order = False

    if field == 'dx':
        data = -atoms1.positions[:,0] + atoms2.positions[:,0]
    elif field == 'dy':
        data = -atoms1.positions[:,1] + atoms2.positions[:,1]
    elif field == 'dz':
        data = -atoms1.positions[:,2] + atoms2.positions[:,2]
    elif field == 'd':
        data = np.linalg.norm(atoms1.positions - atoms2.positions,axis=1)
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
        data = -images1.get_forces(atoms1)[:,0] + images2.get_forces(atoms2)[:,0]
    elif field == 'dfy':
        data = -images1.get_forces(atoms1)[:,1] + images2.get_forces(atoms2)[:,1]
    elif field == 'dfz':
        data = -images1.get_forces(atoms1)[:,2] + images2.get_forces(atoms2)[:,2]
    elif field == 'df':
        data = np.linalg.norm(images1.get_forces(atoms1) - images2.get_forces(atoms2),axis=1)
    elif field == 'afx':
        data = images1.get_forces(atoms1)[:,0] + images2.get_forces(atoms2)[:,0]
        data /= 2
    elif field == 'afy':
        data = images1.get_forces(atoms1)[:,1] + images2.get_forces(atoms2)[:,1]
        data /= 2
    elif field == 'afz':
        data = images1.get_forces(atoms1)[:,2] + images2.get_forces(atoms2)[:,2]
        data /= 2
    elif field == 'af':
        data = np.linalg.norm(images1.get_forces(atoms1) + images2.get_forces(atoms2),axis=1)
        data /=2

    if rank_order: 
        return sort2rank(np.argsort(-data))
    else:
        return data


def format_table(table,fields):
    s = ''.join([fmt[field] for field in fields])
    body = [ formatter(s,*row) for row in table]
    return '\n'.join(body)

def parse_field_specs(field_specs):
    fields=[]; hier = []; scent=[]; 
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
        hier = sort2rank(np.array(hier))[::-1] #reversed by convention of numpy lexsort
    return fields, hier, scent

def render_table(field_specs,atoms1,atoms2,show_only = None):
    fields, hier, scent = parse_field_specs(field_specs)
    fdata = np.array([get_data(atoms1, atoms2, field) for field in fields])
    if hier != []:
        sorting_array = prec_round((np.array(scent)[:,np.newaxis]*fdata)[hier])
        table = fdata[:, np.lexsort(sorting_array)].transpose()
    else:
        table = fdata.transpose()
    if show_only != None:
        table = table[:show_only]
    format_dict['body'] = format_table(table, fields)
    format_dict['header'] = (fmt_class['str']*len(fields)).format(*[header_alias(field) for field in fields])
    format_dict['summary'] = 'RMSD={:+.1E}'.format(np.sqrt(np.power(np.linalg.norm(atoms1.positions-atoms2.positions,axis=1),2).mean()))
    return template.format(**format_dict)

def render_table_calc(field_specs,images1,images2,counter,show_only = None):
    fields, hier, scent = parse_field_specs(field_specs)
    fdata = np.array([get_images_data(images1, images2, counter, field) for field in fields])
    if hier != []:
        sorting_array = prec_round((np.array(scent)[:,np.newaxis]*fdata)[hier])
        table = fdata[:, np.lexsort(sorting_array)].transpose()
    else:
        table = fdata.transpose()
    if show_only != None:
        table = table[:show_only]
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

    It is possible to fully customize the order of the columns and the sorting of the rows by their column fields. A copy of the template file can be placed in ~/.ase where ~ is the user root directory.
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
            type=str, help="""order atoms by rank, see --template help for possible fields
            The default value, when specified, is d. 
            When not specified, ordering is the same as that provided by the generator. 
            For hiearchical sorting, see template.
            """)#,formatter_class=RawTextHelpFormatter)
        add('-c', '--calculator-outputs', action="store_true",
            help="display calculator outputs of forces and energy")
        add('-s', '--show-only', metavar='n', type=int, help="show only so many lines (atoms) in each table, useful if rank ordering")
        add('-t', '--template', metavar='template',nargs='?', const='rc', 
        help="""
        Without argument, looks for ~/.ase/template.py.
        Otherwise, expects the comma separated list of the fields to include in their left-to-right order.
        Optionally, specify the lexicographical sort hierarchy (0 is outermost sort) and if the sort should be ascending or descending (1 or -1).
        By default, sorting is descending, which makes sense for most things except index (and rank, but one can just sort by the thing which is ranked to get ascending ranks).

        example: ase diff start.cif stop.cif --template i:0:1,el,dx,dy,dz,d,rd

        possible fields:
            i: index
            dx,dy,dz,d: displacement/displacement components
            dfx,dfy,dfz,df: difference force/force components
            afx,afy,afz,af: average force/force components (between first and second iteration)
            an: atomic number
            el: atomic element
            t: atom tag
            r<col>: the rank of that atom with respect to the column

        It is possible to change formatters in a template file.
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
            from ase.cli.template import field_specs_on_conditions
            field_specs = field_specs_on_conditions(args.calculator_outputs,args.rank_order)
        elif args.template == 'rc':
            import os
            homedir = os.environ['HOME']
            sys.path.insert(0,homedir+'/.ase')
            from templaterc import format_dict, format_dict_calc, fmt, \
                    fmt_class, header_alias, template, \
                    field_specs_on_conditions
            #this has to be named differently because python does not redundantly load 
            field_specs = field_specs_on_conditions(args.calculator_outputs,args.rank_order)
        else:
            field_specs = args.template.split(',')
            if args.calculator_outputs == False:
                for field_spec in field_specs:
                    if 'f' in field_spec:
                        raise Exception('field requiring calculation outputs without --calculator-outputs')

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
                print('images {}-{}'.format(counter+1,counter))
                t = render_table(field_specs,atoms1,atoms2,show_only = args.show_only)
                print(t,file=out)
            else:
                if args.calculator_outputs == True:
                    for counter in range(len(images1)):
                        print('images {}-{}'.format(counter+1,counter),file=out)
                        t = render_table_calc(field_specs,images1,images2,counter,show_only=args.show_only)
                        print(t,file=out)
                else:
                    for counter in range(len(images1)):
                        print('images {}-{}'.format(counter+1,counter),file=out)
                        t = render_table(field_specs,images1.get_atoms(counter),images2.get_atoms(counter),show_only=args.show_only)
                        print(t,file=out)

        elif len(args.files) == 1:
            f1 = args.files[0]

            if args.calculator_outputs:
                images = Images()
                images.read([f1])
                for counter in range(len(images) - 1):
                    print('images {}-{}'.format(counter+1,counter),file=out)
                    t = render_table_calc(field_specs,images,images,counter,show_only=args.show_only)
                    print(t,file=out)
            else:
                traj = iread(f1)
                atoms_list = list(traj)
                for counter in range(len(atoms_list)-1):
                    t = render_table(field_specs,atoms_list[counter],atoms_list[counter+1],show_only=args.show_only)
                    print('images {}-{}'.format(counter+1,counter),file=out)
                    print(t,file=out)
