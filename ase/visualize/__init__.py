from io import BytesIO
import os
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
        suffix = '.' + self.ioformat.extensions[0]
        return tempfile.NamedTemporaryFile(suffix=suffix)

    def view_blocking(self, atoms, data=None):
        with self.mktemp() as fd:
            if data is None:
                write(fd, atoms, format=self.fmt)
            else:
                write(fd, atoms, format=self.fmt, data=data)
            self.execute_viewer(self.argv + [fd.name])

    def _execute(self, argv):
        subprocess.check_call(argv)

    def view(self, atoms, data=None, repeat=None):
        """Spawn a new process in which to open the viewer."""
        if repeat is not None:
            atoms = atoms.repeat(repeat)

        proc = Popen([sys.executable, '-m', 'ase.visualize.external'],
                     stdin=PIPE)

        pickle.dump((self, atoms, data), proc.stdin)
        proc.stdin.close()
        return proc

    @classmethod
    def viewers(cls):
        return [
            cls('ase_gui_cli', 'traj', [sys.executable, '-m', 'ase.gui']),
            cls('avogadro', 'cube', ['avogadro']),
            cls('gopenmol', 'xyz', ['runGOpenMol']),
            cls('rasmol', 'proteindatabank', ['rasmol', '-pdb']),
            cls('vmd', 'cube', ['vmd']),
            cls('xmakemol', 'xyz', ['xmakemol', '-f']),
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

def view(atoms, data=None, viewer='ase', repeat=None, block=False):
    # Ignore for parallel calculations:
    if parallel.world.size != 1:
        return

    vwr = viewers[viewer.lower()]
    handle = vwr.view(atoms, data=data, repeat=repeat)

    if block and hasattr(handle, 'wait'):
        status = handle.wait()
        if status != 0:
            raise RuntimeError(f'Viewer "{vwr.name}" failed with status '
                               '{status}')

    return handle

def paraview_stuff():
        # macro for showing atoms in paraview
        macro = """\
from paraview.simple import *
version_major = servermanager.vtkSMProxyManager.GetVersionMajor()
source = GetActiveSource()
renderView1 = GetRenderView()
atoms = Glyph(Input=source,
              GlyphType='Sphere',
#              GlyphMode='All Points',
              Scalars='radii',
              ScaleMode='scalar',
              )
RenameSource('Atoms', atoms)
atomsDisplay = Show(atoms, renderView1)
if version_major <= 4:
    atoms.SetScaleFactor = 0.8
    atomicnumbers_PVLookupTable = GetLookupTableForArray( "atomic numbers", 1)
    atomsDisplay.ColorArrayName = ('POINT_DATA', 'atomic numbers')
    atomsDisplay.LookupTable = atomicnumbers_PVLookupTable
else:
    atoms.ScaleFactor = 0.8
    ColorBy(atomsDisplay, 'atomic numbers')
    atomsDisplay.SetScalarBarVisibility(renderView1, True)
Render()
        """
        script_name = os.path.join(tempfile.gettempdir(), 'draw_atoms.py')
        with open(script_name, 'w') as fd:
            fd.write(macro)
        format = 'vtu'
        command = 'paraview --script=' + script_name

    #viewer_obj = viewers[vwr]
    #process = viewer_obj.view(atoms, data=data)


def main():
    cli_viewer, atoms, data = pickle.load(sys.stdin.buffer)
    cli_viewer.view_blocking(atoms, data)


if __name__ == '__main__':
    main()
