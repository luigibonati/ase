from io import BytesIO
import os
import pickle
import subprocess
import sys
import tempfile

from ase.io.formats import ioformats
from ase.io import write
import ase.parallel as parallel


def _pipe_to_ase_gui(atoms, repeat):
    buf = BytesIO()
    write(buf, atoms, format='traj')

    args = [sys.executable, '-m', 'ase', 'gui', '-']
    if repeat:
        args.append('--repeat={},{},{}'.format(*repeat))

    proc = subprocess.Popen(args, stdin=subprocess.PIPE)
    proc.stdin.write(buf.getvalue())
    proc.stdin.close()
    return proc


class CLIViewer:
    def __init__(self, name, fmt, argv):
        self.name = name
        self.fmt = fmt
        self.argv = argv

    @property
    def ioformat(self):
        return ioformats[self.fmt]

    def mktemp(self):
        ioformat = self.ioformat
        suffix = '.' + ioformat.extensions[0]
        if ioformat.isbinary:
            mode = 'wb'
        else:
            mode = 'w'
        return tempfile.NamedTemporaryFile(mode=mode, suffix=suffix)

    def view_blocking(self, atoms, data=None):
        with self.mktemp() as fd:
            if data is None:
                write(fd, atoms, format=self.fmt)
            else:
                write(fd, atoms, format=self.fmt, data=data)
            self.execute_viewer(fd.name)

    def execute_viewer(self, filename):
        print('GRRR', repr(self.argv + [filename]))
        subprocess.check_call(self.argv + [filename])

    def view(self, atoms, data=None, repeat=None):
        """Spawn a new process in which to open the viewer."""
        if repeat is not None:
            atoms = atoms.repeat(repeat)

        proc = subprocess.Popen(
            [sys.executable, '-m', 'ase.visualize.external'],
            stdin=subprocess.PIPE)

        pickle.dump((self, atoms, data), proc.stdin)
        proc.stdin.close()
        return proc

    @classmethod
    def viewers(cls):
        return [
            cls('ase_gui_cli', 'traj', [sys.executable, '-m', 'ase.gui']),
            cls('avogadro', 'cube', ['avogadro']),
            cls('gopenmol', 'extxyz', ['runGOpenMol']),
            cls('rasmol', 'proteindatabank', ['rasmol', '-pdb']),
            cls('vmd', 'cube', ['vmd']),
            cls('xmakemol', 'extxyz', ['xmakemol', '-f']),
        ]


class PyViewer:
    def __init__(self, name, supports_repeat=False):
        self.name = name
        self.supports_repeat = supports_repeat

    def view(self, atoms, data=None, repeat=None):
        # Delegate to any of the below methods
        func = getattr(self, self.name)
        if self.supports_repeat:
            return func(atoms, repeat)
        else:
            if repeat is not None:
                atoms = atoms.repeat(repeat)
            return func(atoms)

    def sage(self, atoms):
        from ase.visualize.sage import view_sage_jmol
        return view_sage_jmol(atoms)

    def ngl(self, atoms):
        from ase.visualize.ngl import view_ngl
        return view_ngl(atoms)

    def x3d(self, atoms):
        from ase.visualize.x3d import view_x3d
        return view_x3d(atoms)

    def ase(self, atoms, repeat):
        return _pipe_to_ase_gui(atoms, repeat)

    @classmethod
    def viewers(cls):
        return [
            cls('ase', supports_repeat=True),
            cls('ngl'),
            cls('sage'),
            cls('x3d'),
        ]



viewers = {viewer.name: viewer
           for viewer in CLIViewer.viewers() + PyViewer.viewers()}
viewers['nglview'] = viewers['ngl']


def main():
    cli_viewer, atoms, data = pickle.load(sys.stdin.buffer)
    cli_viewer.view_blocking(atoms, data)


if __name__ == '__main__':
    main()
