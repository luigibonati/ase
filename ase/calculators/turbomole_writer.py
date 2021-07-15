"""Module containing code to manupulate control file"""
from ase.calculators.turbomole_executor import execute


def add_data_group(data_group, string=None, raw=False):
    """write a turbomole data group to control file"""
    if raw:
        data = data_group
    else:
        data = '$' + data_group
        if string:
            data += ' ' + string
        data += '\n'
    f = open('control', 'r+')
    lines = f.readlines()
    f.seek(0)
    f.truncate()
    lines.insert(2, data)
    f.write(''.join(lines))
    f.close()


def delete_data_group(data_group):
    """delete a turbomole data group from control file"""
    command = ['kdg', data_group]
    execute(command, error_test=False, stdout_tofile=False)
